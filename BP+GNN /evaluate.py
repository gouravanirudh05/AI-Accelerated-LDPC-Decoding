"""BER/BLER vs Eb/N0 harness for BP and the graph correctors.

    python -m ldpc_corrector.evaluate \\
        --checkpoint Results_Corrector/gcn.best.pt \\
        --checkpoint Results_Corrector/gat.best.pt \\
        --bp-iterations 5 20 --out-dir Results_Corrector/eval

Sweeps every requested variant over the same seeded frames, writes a tidy CSV,
overlays the already-reproduced ECCT/AECCT numbers, and reports compute per
decode -- FLOPs *and* wall-clock -- because the whole point of a corrector on
top of BP(5) is the cost/accuracy tradeoff against BP(20).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from ._compat import Get_Generator_and_Parity, sample_channel_batch, set_seed
from .config import RunConfig, add_config_arguments, load_config
from .decoder import BPBaseline, CorrectorDecoder, LegacyGATBPDecoder, decode_bits
from .train import apply_thread_setting, build_code, build_model, resolve_device

LOGGER = logging.getLogger("ldpc_corrector.evaluate")

CSV_FIELDS = [
    "model",
    "ebno_db",
    "ber",
    "fer",
    "neg_ln_ber",
    "frames",
    "bit_errors",
    "frame_errors",
    "flops_per_frame",
    "seconds_per_frame",
    "params",
    "paired_broke",
    "paired_fixed",
    "paired_net",
    "paired_z",
]


def configure_logger() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )


# ----------------------------------------------------------------- sweeping


@torch.no_grad()
def sweep_decoder(
    decoder: nn.Module,
    generator: Tensor,
    ebn0_points: Sequence[float],
    frames: int,
    batch_size: int,
    device: torch.device,
    min_frame_errors: int = 0,
    seed: int = 1234,
    collect_failures: bool = False,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[float, Tensor]]:
    """BER/FER at each Eb/N0 point, on seeded frames shared across variants.

    With ``collect_failures`` the per-frame failure mask is returned too, for a
    paired (McNemar) comparison against a reference decoder.  Early stopping is
    then disabled: pairing requires every variant to decode the *same* frames,
    and ``min_frame_errors`` would cut each variant off at a different point.
    """
    decoder.eval()
    rows: List[Dict[str, Any]] = []
    failures: Dict[float, Tensor] = {}
    if collect_failures:
        min_frame_errors = 0

    for index, ebn0 in enumerate(ebn0_points):
        bit_errors = frame_errors = 0
        total_bits = total_frames = 0
        frame_failed: List[Tensor] = []
        # Same seed per SNR point for every decoder, so all variants see
        # identical channel realizations.
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed + index)

        while total_frames < frames:
            current = min(batch_size, frames - total_frames)
            channel_llr, codewords, _, _ = sample_channel_batch(
                generator.cpu(), current, float(ebn0), torch.device("cpu"), rng=rng
            )
            final_llr = decoder(channel_llr.to(device))
            errors = decode_bits(final_llr) != codewords.to(device)

            failed = errors.any(dim=1)
            if collect_failures:
                frame_failed.append(failed.cpu())
            bit_errors += int(errors.sum())
            frame_errors += int(failed.sum())
            total_bits += errors.numel()
            total_frames += current

            # Enough frame errors for a stable estimate: stop paying for more.
            if min_frame_errors and frame_errors >= min_frame_errors:
                break

        ber = bit_errors / max(total_bits, 1)
        rows.append(
            {
                "ebno_db": float(ebn0),
                "ber": ber,
                "fer": frame_errors / max(total_frames, 1),
                "neg_ln_ber": float(-np.log(ber)) if ber > 0 else float("inf"),
                "frames": total_frames,
                "bit_errors": bit_errors,
                "frame_errors": frame_errors,
            }
        )
        if collect_failures:
            failures[float(ebn0)] = torch.cat(frame_failed)
        LOGGER.info(
            "  %5.2f dB | ber %.3e fer %.3e (%d frames, %d frame errors)",
            ebn0,
            rows[-1]["ber"],
            rows[-1]["fer"],
            total_frames,
            frame_errors,
        )
    if collect_failures:
        return rows, failures
    return rows


@torch.no_grad()
def measure_seconds_per_frame(
    decoder: nn.Module,
    generator: Tensor,
    device: torch.device,
    batch_size: int = 512,
    batches: int = 5,
    ebn0: float = 4.0,
) -> float:
    """Wall-clock per decoded frame, as an empirical check on the FLOP model."""
    decoder.eval()
    channel_llr, _, _, _ = sample_channel_batch(
        generator.cpu(), batch_size, ebn0, torch.device("cpu")
    )
    channel_llr = channel_llr.to(device)

    decoder(channel_llr)  # warm up allocators / kernel selection
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(batches):
        decoder(channel_llr)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / (batches * batch_size)


# -------------------------------------------------------------- assembling


def load_checkpoint_decoder(
    path: Path,
    parity_check: np.ndarray,
    device: torch.device,
    resume_iterations: Optional[int] = None,
) -> Tuple[str, CorrectorDecoder]:
    """Rebuild a trained corrector from the config stored in its checkpoint.

    ``resume_iterations`` overrides the stored ``bp.resume_iterations``, to
    run a corrector trained without resumed BP *with* it (no new weights: the
    resumed BP has none).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stored = payload.get("config")
    if stored is None:
        raise ValueError(f"{path} has no embedded config; cannot rebuild the model.")
    config = RunConfig(**_rehydrate(stored))
    if resume_iterations is not None:
        config.bp.resume_iterations = int(resume_iterations)

    model = build_model(config, parity_check)
    incompatible = model.load_state_dict(payload["model_state"], strict=False)
    # The resumed BP holds only index buffers rebuilt from H, never trained
    # weights, so its keys may be absent (or extra) when resume is overridden.
    # Any other mismatch is a genuinely wrong checkpoint.
    stray = [
        key
        for key in (*incompatible.missing_keys, *incompatible.unexpected_keys)
        if not key.startswith("resume_bp.")
    ]
    if stray:
        raise RuntimeError(f"{path} does not match its own config: {stray}")
    model.to(device)
    label = f"{model.name}[{config.name}]"
    return label, model


def _rehydrate(stored: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a serialized RunConfig dict back into dataclass sections."""
    from .config import BPConfig, CodeConfig, EvalConfig, ModelConfig, TrainConfig

    sections = {
        "code": CodeConfig,
        "bp": BPConfig,
        "model": ModelConfig,
        "train": TrainConfig,
        "eval": EvalConfig,
    }
    rebuilt = dict(stored)
    for key, cls in sections.items():
        if isinstance(rebuilt.get(key), dict):
            rebuilt[key] = cls(**rebuilt[key])
    return rebuilt


def read_reference_csv(path: Path, models: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Load already-reproduced baselines (ECCT/AECCT) from a results CSV."""
    if not path.is_file():
        LOGGER.warning("reference CSV %s not found; skipping overlay", path)
        return []

    rows: List[Dict[str, Any]] = []
    with path.open() as handle:
        for record in csv.DictReader(handle):
            name = record.get("model", "").strip()
            if not name or name == "model":
                continue
            if models and name not in models:
                continue
            try:
                ber = float(record["ber"])
                rows.append(
                    {
                        "model": name,
                        "ebno_db": float(record["ebno_db"]),
                        "ber": ber,
                        "fer": float(record["fer"]),
                        "neg_ln_ber": float(-np.log(ber)) if ber > 0 else float("inf"),
                        "frames": int(float(record.get("n_frames") or 0)),
                        "bit_errors": "",
                        "frame_errors": int(float(record.get("n_frame_errors") or 0)),
                        "flops_per_frame": "",
                        "seconds_per_frame": "",
                        "params": "",
                    }
                )
            except (KeyError, ValueError) as error:
                LOGGER.warning("skipping malformed row in %s: %s", path, error)
    LOGGER.info("overlaid %d reference rows from %s", len(rows), path)
    return rows


def mcnemar(reference_failed: Tensor, variant_failed: Tensor) -> Dict[str, float]:
    """Paired comparison of two decoders on identical frames.

    Only the frames where they disagree carry information: ``b`` frames the
    reference decoded and the variant did not, ``c`` the reverse.  Under the
    null the count difference has variance ``b + c``, so ``z = (c - b) /
    sqrt(b + c)``; positive ``z`` favours the variant.

    Comparing marginal error rates instead would understate significance, since
    the shared channel noise dominates both and cancels in the pairing.
    """
    if reference_failed.shape != variant_failed.shape:
        raise ValueError(
            "Paired comparison needs equal frame counts; got "
            f"{tuple(reference_failed.shape)} vs {tuple(variant_failed.shape)}."
        )
    broke = int((~reference_failed & variant_failed).sum())
    fixed = int((reference_failed & ~variant_failed).sum())
    discordant = broke + fixed
    z = (fixed - broke) / math.sqrt(discordant) if discordant else 0.0
    return {
        "paired_broke": broke,
        "paired_fixed": fixed,
        "paired_net": fixed - broke,
        "paired_z": z,
    }


def write_csv(
    rows: Sequence[Dict[str, Any]], path: Path, extra_fields: Sequence[str] = ()
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(CSV_FIELDS) + [f for f in extra_fields if f not in CSV_FIELDS]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    LOGGER.info("wrote %s (%d rows)", path, len(rows))


def plot_curves(rows: Sequence[Dict[str, Any]], out_dir: Path, tag: str = "") -> List[Path]:
    """BER and BLER vs Eb/N0, every variant overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    written: List[Path] = []
    for metric, ylabel, filename in (
        ("ber", "Bit error rate", "ber_vs_ebno"),
        ("fer", "Block error rate", "bler_vs_ebno"),
    ):
        figure, axis = plt.subplots(figsize=(7.5, 5.5))
        for name in sorted(by_model):
            points = sorted(by_model[name], key=lambda r: r["ebno_db"])
            x = [p["ebno_db"] for p in points]
            y = [p[metric] for p in points]
            # A zero rate cannot be drawn on a log axis; drop those points and
            # let the curve end rather than silently plotting a floor.
            visible = [(a, b) for a, b in zip(x, y) if b and b > 0]
            if not visible:
                continue
            is_reference = name.startswith(("ECCT", "AECCT"))
            axis.semilogy(
                [a for a, _ in visible],
                [b for _, b in visible],
                marker="x" if name.startswith("BP(") else "o",
                linestyle="--" if name.startswith("BP(") or is_reference else "-",
                linewidth=1.6,
                markersize=5,
                label=name,
            )
        axis.set_xlabel("$E_b/N_0$ (dB)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} vs $E_b/N_0$ — LDPC(49, 24)")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()

        path = out_dir / f"{filename}{tag}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(path)
        LOGGER.info("wrote %s", path)
    return written


# ------------------------------------------------------------------- driver


def evaluate(
    config: RunConfig,
    checkpoints: Sequence[Path],
    bp_iterations: Sequence[int],
    out_dir: Path,
    reference_csv: Optional[Path] = None,
    reference_models: Optional[Sequence[str]] = None,
    legacy_checkpoints: Optional[Sequence[Path]] = None,
    resume_checkpoints: Optional[Sequence[Path]] = None,
    resume_iterations: int = 0,
    extra_references: Sequence[str] = (),
) -> Path:
    configure_logger()
    set_seed(config.seed, config.deterministic)
    apply_thread_setting(config.torch_threads)
    device = resolve_device(config.device)

    generator, parity_check = build_code(config)
    generator = generator.to(device)

    variants: List[Tuple[str, nn.Module]] = [
        (f"BP({iterations})", BPBaseline(parity_check, iterations).to(device))
        for iterations in bp_iterations
    ]
    for path in checkpoints:
        variants.append(load_checkpoint_decoder(Path(path), parity_check, device))
    for path in resume_checkpoints or []:
        variants.append(
            load_checkpoint_decoder(Path(path), parity_check, device, resume_iterations)
        )
    for path in legacy_checkpoints or []:
        legacy = LegacyGATBPDecoder(path, parity_check).to(device)
        LOGGER.info(
            "loaded %s (recorded validation FER %s)",
            legacy.name,
            legacy.recorded_validation_fer,
        )
        variants.append((legacy.name, legacy))

    rows: List[Dict[str, Any]] = []
    cost_summary: List[Dict[str, Any]] = []
    # Per-frame failure masks, so each variant can be compared against the
    # reference decoder on identical frames rather than by marginal rates.
    failure_masks: Dict[str, Dict[float, Tensor]] = {}
    paired_reference = f"BP({bp_iterations[0]})" if bp_iterations else None

    for label, decoder in variants:
        LOGGER.info("sweeping %s", label)
        seconds = measure_seconds_per_frame(
            decoder,
            generator,
            device,
            batches=config.eval.timing_batches,
        )
        flops = float(decoder.estimate_flops_per_frame())
        params = (
            decoder.corrector.num_parameters()
            if isinstance(decoder, CorrectorDecoder)
            else 0
        )
        swept, masks = sweep_decoder(
            decoder,
            generator,
            config.eval.ebn0,
            config.eval.frames,
            config.eval.batch_size,
            device,
            min_frame_errors=config.eval.min_frame_errors,
            seed=config.validation_seed,
            collect_failures=True,
        )
        failure_masks[label] = masks
        for row in swept:
            rows.append(
                {
                    "model": label,
                    **row,
                    "flops_per_frame": flops,
                    "seconds_per_frame": seconds,
                    "params": params,
                }
            )
        cost_summary.append(
            {
                "model": label,
                "flops_per_frame": flops,
                "seconds_per_frame": seconds,
                "params": params,
            }
        )
        LOGGER.info(
            "  cost | %.0f FLOPs/frame | %.2f us/frame | %d corrector params",
            flops,
            seconds * 1e6,
            params,
        )

    # Paired comparison against the reference decoder, per SNR point.
    if paired_reference and paired_reference in failure_masks:
        reference_masks = failure_masks[paired_reference]
        for row in rows:
            label, snr = row["model"], row.get("ebno_db")
            if label == paired_reference or label not in failure_masks:
                continue
            variant_mask = failure_masks[label].get(snr)
            reference_mask = reference_masks.get(snr)
            if variant_mask is None or reference_mask is None:
                continue
            row.update(mcnemar(reference_mask, variant_mask))

        LOGGER.info("paired comparison against %s (z>1.96 favours the variant):", paired_reference)
        for label in failure_masks:
            if label == paired_reference:
                continue
            for snr in sorted(reference_masks):
                stats = mcnemar(reference_masks[snr], failure_masks[label][snr])
                LOGGER.info(
                    "  %-28s %4.1f dB | fixed %5d broke %5d net %+5d z %+6.2f",
                    label,
                    snr,
                    stats["paired_fixed"],
                    stats["paired_broke"],
                    stats["paired_net"],
                    stats["paired_z"],
                )

    # Further paired references, e.g. BP(10): same frames, so McNemar applies.
    extra_fields: List[str] = []
    for reference in extra_references:
        if reference not in failure_masks:
            LOGGER.warning("extra reference %s was not swept; skipped", reference)
            continue
        tag = reference.replace("(", "").replace(")", "").lower()
        extra_fields += [f"paired_fixed_vs_{tag}", f"paired_broke_vs_{tag}", f"paired_z_vs_{tag}"]
        LOGGER.info("paired comparison against %s:", reference)
        for row in rows:
            label, snr = row["model"], row.get("ebno_db")
            if label == reference or label not in failure_masks:
                continue
            stats = mcnemar(failure_masks[reference][snr], failure_masks[label][snr])
            row[f"paired_fixed_vs_{tag}"] = stats["paired_fixed"]
            row[f"paired_broke_vs_{tag}"] = stats["paired_broke"]
            row[f"paired_z_vs_{tag}"] = stats["paired_z"]
            LOGGER.info("  %-44s %4.1f dB | fixed %5d broke %5d z %+6.2f",
                        label[:44], snr, stats["paired_fixed"], stats["paired_broke"],
                        stats["paired_z"])

    if reference_csv is not None:
        rows.extend(read_reference_csv(Path(reference_csv), reference_models))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    write_csv(rows, csv_path, extra_fields)
    (out_dir / "cost.json").write_text(json.dumps(cost_summary, indent=2))
    plot_curves(rows, out_dir)

    LOGGER.info("compute per decode:")
    baseline = next(
        (c for c in cost_summary if c["model"] == "BP(20)"), None
    )
    for entry in cost_summary:
        relative = (
            f" ({entry['flops_per_frame'] / baseline['flops_per_frame']:.2f}x BP(20))"
            if baseline and baseline["flops_per_frame"]
            else ""
        )
        LOGGER.info(
            "  %-28s %12.0f FLOPs/frame  %8.2f us/frame%s",
            entry["model"],
            entry["flops_per_frame"],
            entry["seconds_per_frame"] * 1e6,
            relative,
        )
    return csv_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Trained corrector checkpoint; repeatable.",
    )
    parser.add_argument(
        "--bp-iterations",
        type=int,
        nargs="*",
        default=[5, 20],
        help="BP-only reference curves.",
    )
    parser.add_argument("--out-dir", default="Results_Corrector/eval")
    parser.add_argument(
        "--reference-csv",
        default="comparison/data/test_results.csv",
        help="CSV of already-reproduced ECCT/AECCT results to overlay.",
    )
    parser.add_argument(
        "--reference-models",
        nargs="*",
        default=None,
        help="Subset of reference CSV model names to overlay.",
    )
    parser.add_argument(
        "--no-reference", action="store_true", help="Skip the ECCT/AECCT overlay."
    )
    parser.add_argument(
        "--resume-checkpoint",
        action="append",
        default=[],
        help="Corrector checkpoint to sweep with --resume-iterations of BP "
        "resumed after its correction (no retraining); repeatable.",
    )
    parser.add_argument("--resume-iterations", type=int, default=5)
    parser.add_argument(
        "--paired-against",
        nargs="*",
        default=[],
        metavar="REF",
        help="Extra paired references besides the first BP, e.g. 'BP(10)'.",
    )
    parser.add_argument(
        "--legacy-checkpoint",
        action="append",
        default=[],
        help="Phase 0-2 HybridBPGAT checkpoint (e.g. GAT+BP/best_model.pt); repeatable.",
    )
    args = parser.parse_args(argv)

    evaluate(
        load_config(args.config, args.overrides),
        [Path(p) for p in args.checkpoint],
        args.bp_iterations,
        Path(args.out_dir),
        None if args.no_reference else Path(args.reference_csv),
        args.reference_models,
        [Path(p) for p in args.legacy_checkpoint],
        [Path(p) for p in args.resume_checkpoint],
        args.resume_iterations,
        args.paired_against,
    )


if __name__ == "__main__":
    main()
