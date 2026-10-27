"""Run the ablation sweep described by a sweep YAML file.

    python -m ldpc_corrector.ablate --sweep ldpc_corrector/configs/ablations.yaml
    python -m ldpc_corrector.ablate --dry-run
    python -m ldpc_corrector.ablate --only gat_excl_on gat_excl_off

Each variant is a base config plus dotted overrides, so the sweep adds no new
code paths.  Results land in one CSV alongside the checkpoints, and any variant
that fails is reported rather than aborting the whole sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from .config import load_config
from .train import train

LOGGER = logging.getLogger("ldpc_corrector.ablate")

DEFAULT_SWEEP = Path(__file__).resolve().parent / "configs" / "ablations.yaml"
SUMMARY_FIELDS = [
    "name",
    "corrector",
    "status",
    "best_validation_fer",
    "params",
    "flops_per_frame",
    "train_seconds",
    "checkpoint",
    "overrides",
]


def load_sweep(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text()) or {}
    if "variants" not in payload:
        raise ValueError(f"{path} must define a 'variants' list.")
    return payload


def resolve_variant(
    variant: Dict[str, Any], common: Sequence[str], config_dir: Path
) -> Any:
    """Build the RunConfig for one variant: base + common + own overrides."""
    name = variant.get("name")
    if not name:
        raise ValueError(f"Every variant needs a name: {variant!r}")
    base = variant.get("base")
    base_path = (config_dir / base) if base else None
    overrides = list(common) + list(variant.get("overrides") or [])
    # The variant name is the run name, so checkpoints do not collide.
    overrides.append(f"name={name}")
    return load_config(base_path, overrides)


def run_sweep(
    sweep_path: Path,
    only: Optional[Sequence[str]] = None,
    dry_run: bool = False,
) -> Optional[Path]:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    sweep = load_sweep(sweep_path)
    common = sweep.get("common") or []
    config_dir = sweep_path.parent

    variants = sweep["variants"]
    if only:
        by_name = {v.get("name"): v for v in variants}
        unknown = [name for name in only if name not in by_name]
        if unknown:
            raise KeyError(f"Unknown variant names: {sorted(unknown)}")
        # Run in the order given rather than the order they appear in the file:
        # --only is used as a priority list, and a long sweep may be cut short,
        # so the most informative variants have to go first.
        ordered, seen = [], set()
        for name in only:
            if name not in seen:
                seen.add(name)
                ordered.append(by_name[name])
        variants = ordered

    LOGGER.info("%d variant(s) to run from %s", len(variants), sweep_path)
    rows: List[Dict[str, Any]] = []
    save_dir: Optional[Path] = None

    for index, variant in enumerate(variants, 1):
        config = resolve_variant(variant, common, config_dir)
        save_dir = Path(config.save_dir)
        overrides = " ".join(list(common) + list(variant.get("overrides") or []))

        if dry_run:
            LOGGER.info(
                "[%d/%d] %s | corrector=%s | %s",
                index,
                len(variants),
                config.name,
                config.model.corrector,
                overrides,
            )
            continue

        LOGGER.info("[%d/%d] training %s", index, len(variants), config.name)
        started = time.time()
        row: Dict[str, Any] = {
            "name": config.name,
            "corrector": config.model.corrector,
            "overrides": overrides,
        }
        try:
            checkpoint = train(config)
            import torch

            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            row.update(
                status="ok",
                best_validation_fer=payload.get("validation_fer"),
                checkpoint=str(checkpoint),
            )
        except Exception as error:  # keep the sweep going
            LOGGER.error("variant %s failed: %s", config.name, error)
            LOGGER.debug("%s", traceback.format_exc())
            row.update(status=f"failed: {error}", best_validation_fer="", checkpoint="")

        row["train_seconds"] = round(time.time() - started, 1)
        rows.append(row)

    if dry_run or save_dir is None:
        return None

    save_dir.mkdir(parents=True, exist_ok=True)
    summary = save_dir / "ablation_summary.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    LOGGER.info("wrote %s", summary)

    ranked = sorted(
        (r for r in rows if r.get("status") == "ok"),
        key=lambda r: r.get("best_validation_fer") or float("inf"),
    )
    LOGGER.info("ranked by validation FER:")
    for row in ranked:
        LOGGER.info("  %-24s %.6f", row["name"], row["best_validation_fer"])
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default=str(DEFAULT_SWEEP))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run_sweep(Path(args.sweep), args.only, args.dry_run)


if __name__ == "__main__":
    main()
