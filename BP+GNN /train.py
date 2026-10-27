"""Train a graph residual corrector on top of a frozen truncated BP(k).

    python -m ldpc_corrector.train --config ldpc_corrector/configs/gcn.yaml

Per batch: sample mixed-SNR AWGN frames -> run frozen BP(k) -> derive trace
features -> corrector forward -> combine -> loss.  Each loss term is logged
separately, because a total alone cannot tell you which term is doing work.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ._compat import Get_Generator_and_Parity, sample_channel_batch, set_seed
from .config import RunConfig, add_config_arguments, load_config
from .decoder import CorrectorDecoder, decode_bits
from .losses import build_loss
from .models.base_corrector import combine

LOGGER = logging.getLogger("ldpc_corrector.train")


# ------------------------------------------------------------------- setup


def configure_logger(log_file: Optional[Path]) -> logging.Logger:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    # The ablation runner installs a root handler; without this, every record
    # would be printed twice.
    LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
    return LOGGER


def resolve_device(requested: str) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_thread_setting(threads: int) -> None:
    """Pin intra-op threads (see ``RunConfig.torch_threads`` for why)."""
    if threads > 0:
        torch.set_num_threads(threads)


def build_code(config: RunConfig) -> Tuple[Tensor, np.ndarray]:
    """Load the generator and parity-check matrices for the configured code."""

    class _Code:
        n = config.code.n
        k = config.code.k
        code_type = config.code.code_type

    generator, parity_check = Get_Generator_and_Parity(_Code())
    return torch.as_tensor(generator, dtype=torch.float32), np.asarray(parity_check)


def build_model(config: RunConfig, parity_check: np.ndarray) -> CorrectorDecoder:
    model_config = config.model
    corrector_kwargs: Dict[str, Any] = {
        "hidden_dim": model_config.hidden_dim,
        "dropout": model_config.dropout,
        "correction_scale": model_config.correction_scale,
        "use_gate": model_config.use_gate,
        **model_config.extra,
    }
    return CorrectorDecoder(
        parity_check,
        corrector=model_config.corrector,
        bp_iterations=config.bp.iterations,
        bp_damping=config.bp.damping,
        freeze_bp=config.bp.freeze,
        fixed_gate=model_config.fixed_gate,
        corrector_kwargs=corrector_kwargs,
        resume_iterations=config.bp.resume_iterations,
    )


# ------------------------------------------------------------------ sampling


def sample_ebn0(config: RunConfig, batch_size: int, device: torch.device) -> Tensor:
    """Mixed-SNR sampling: one Eb/N0 per frame in the batch."""
    train_config = config.train
    distribution = train_config.ebn0_distribution.lower()
    if distribution == "uniform":
        span = train_config.ebn0_max - train_config.ebn0_min
        return train_config.ebn0_min + span * torch.rand(batch_size, device=device)
    if distribution == "discrete":
        points = torch.tensor(config.eval.ebn0, device=device, dtype=torch.float32)
        picks = torch.randint(0, points.numel(), (batch_size,), device=device)
        return points[picks]
    raise ValueError(
        f"Unknown ebn0_distribution {train_config.ebn0_distribution!r}; "
        "expected 'uniform' or 'discrete'."
    )


def make_validation_frames(
    config: RunConfig, generator: Tensor, device: torch.device
) -> List[Tuple[float, Tensor, Tensor]]:
    """A fixed, seeded validation set so mid-training numbers are comparable."""
    frames: List[Tuple[float, Tensor, Tensor]] = []
    batch_size = min(config.eval.batch_size, 1024)
    for index, ebn0 in enumerate(config.eval.ebn0):
        rng = torch.Generator(device="cpu")
        rng.manual_seed(config.validation_seed + index)
        channel_llr, codewords, _, _ = sample_channel_batch(
            generator.cpu(),
            batch_size,
            float(ebn0),
            torch.device("cpu"),
            zero_codeword=config.train.zero_codeword,
            rng=rng,
        )
        frames.append((float(ebn0), channel_llr.to(device), codewords.to(device)))
    return frames


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def quick_validate(
    model: CorrectorDecoder, frames: List[Tuple[float, Tensor, Tensor]]
) -> Dict[str, float]:
    """BER/FER on the fixed validation frames, plus the same for BP alone."""
    model.eval()
    metrics: Dict[str, float] = {}
    total_bits = total_bit_errors = 0
    total_frames = total_frame_errors = 0
    bp_frame_errors = 0

    for ebn0, channel_llr, codewords in frames:
        final_llr, aux = model(channel_llr, return_aux=True)
        predicted = decode_bits(final_llr)
        errors = predicted != codewords

        bit_errors = int(errors.sum())
        frame_errors = int(errors.any(dim=1).sum())
        metrics[f"ber@{ebn0:g}"] = bit_errors / errors.numel()
        metrics[f"fer@{ebn0:g}"] = frame_errors / errors.shape[0]

        total_bit_errors += bit_errors
        total_bits += errors.numel()
        total_frame_errors += frame_errors
        total_frames += errors.shape[0]
        bp_frame_errors += int(
            (decode_bits(aux["bp_llr"]) != codewords).any(dim=1).sum()
        )

    metrics["ber"] = total_bit_errors / max(total_bits, 1)
    metrics["fer"] = total_frame_errors / max(total_frames, 1)
    metrics["bp_fer"] = bp_frame_errors / max(total_frames, 1)
    model.train()
    return metrics


# ------------------------------------------------------------------ training


def train(config: RunConfig) -> Path:
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    configure_logger(save_dir / f"{config.name}.log")

    set_seed(config.seed, config.deterministic)
    apply_thread_setting(config.torch_threads)
    device = resolve_device(config.device)

    generator, parity_check = build_code(config)
    generator = generator.to(device)
    model = build_model(config, parity_check).to(device)
    loss_fn = build_loss(config.loss, model.graph.to(device))

    LOGGER.info("Config:\n%s", config.to_json())
    LOGGER.info(
        "%s | corrector params %d | %.0f FLOPs/frame | residual %s, "
        "effective range +/-%.1f LLR",
        model.name,
        model.corrector.num_parameters(),
        model.estimate_flops_per_frame(),
        model.corrector.residual_mode,
        model.corrector.effective_correction_range,
    )
    (save_dir / f"{config.name}.config.json").write_text(config.to_json())

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.train.steps, 1), eta_min=config.train.min_lr
    )

    validation_frames = make_validation_frames(config, generator, device)
    needs_aux = bool(config.loss.get("deep_supervision", False))

    history: List[Dict[str, Any]] = []
    best_fer = math.inf
    best_path = save_dir / f"{config.name}.best.pt"
    running: Dict[str, float] = {}
    running_steps = 0
    start_time = time.time()

    model.train()
    for step in range(1, config.train.steps + 1):
        ebn0 = sample_ebn0(config, config.train.batch_size, device)
        channel_llr, codewords, _, _ = sample_channel_batch(
            generator,
            config.train.batch_size,
            ebn0,
            device,
            zero_codeword=config.train.zero_codeword,
        )

        if needs_aux:
            final_llr, aux = model(channel_llr, return_aux=True)
            stage_llrs = aux["stage_llrs"]
            bp_llr = aux["bp_llr"]
        else:
            bp_llr, trace = model.run_bp(channel_llr)
            vn_features, cn_features, _ = model.features(trace)
            output = model.corrector(
                vn_features,
                cn_features,
                model.graph.vn_to_cn,
                model.graph.cn_to_vn,
                return_gate=True,
            )
            final_llr = combine(
                bp_llr,
                output.delta_llr,
                output.gate_logits,
                fixed_gate=model.corrector.fixed_gate,
            )
            final_llr = model.resume(channel_llr, bp_llr, final_llr, trace)
            stage_llrs = None

        loss, breakdown = loss_fn(final_llr, codewords, bp_llr, stage_llrs)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.train.grad_clip:
            torch.nn.utils.clip_grad_norm_(trainable, config.train.grad_clip)
        optimizer.step()
        scheduler.step()

        for key, value in breakdown.items():
            running[key] = running.get(key, 0.0) + value
        running_steps += 1

        if step % config.train.log_every == 0:
            averages = {k: v / running_steps for k, v in running.items()}
            LOGGER.info(
                "step %d/%d lr %.2e | %s",
                step,
                config.train.steps,
                scheduler.get_last_lr()[0],
                " ".join(f"{k}={v:.5f}" for k, v in sorted(averages.items())),
            )
            history.append({"step": step, "phase": "train", **averages})
            running, running_steps = {}, 0

        if step % config.train.eval_every == 0 or step == config.train.steps:
            metrics = quick_validate(model, validation_frames)
            LOGGER.info(
                "step %d | val ber %.6f fer %.6f (BP-only fer %.6f)",
                step,
                metrics["ber"],
                metrics["fer"],
                metrics["bp_fer"],
            )
            history.append({"step": step, "phase": "val", **metrics})
            if metrics["fer"] < best_fer:
                best_fer = metrics["fer"]
                torch.save(
                    {
                        "step": step,
                        "config": config.to_dict(),
                        "model_state": model.state_dict(),
                        "validation_fer": best_fer,
                    },
                    best_path,
                )
                LOGGER.info("saved %s (val fer %.6f)", best_path.name, best_fer)

    torch.save(
        {
            "step": config.train.steps,
            "config": config.to_dict(),
            "model_state": model.state_dict(),
            "validation_fer": best_fer,
        },
        save_dir / f"{config.name}.last.pt",
    )
    (save_dir / f"{config.name}.history.json").write_text(json.dumps(history, indent=2))
    LOGGER.info(
        "done in %.1fs | best val fer %.6f | %s",
        time.time() - start_time,
        best_fer,
        best_path,
    )
    return best_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
