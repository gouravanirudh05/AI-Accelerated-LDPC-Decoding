"""Package an experiment as a private Kaggle GPU kernel and optionally push it.

    python sparse_gt/kaggle/build_kernel.py --experiment step3 --username gouravanirudh \
        [--token-file ~/.kaggle/access_token_gpu] [--slug-suffix wimax] \
        --args="--code WIMAX_N576_K288 --snr 1.5 2.75" --push

Bundles ``sparse_gt`` (with tests) and ``codes/`` into ``run_experiment.py``,
writes ``kernel-metadata.json`` next to it, and pushes with the Kaggle CLI.
Pass ``--args`` with ``=`` (its value starts with ``--``). Then::

    kaggle kernels status <user>/<slug>
    kaggle kernels output <user>/<slug> -p results/<dir>
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# experiment -> kernel title; Kaggle derives the slug from the title.
TITLES = {"step2": "Sparse GT Step2 BP Coverage", "step3": "Sparse GT Step3 Train",
          "step4": "LDPC GNN Step4 GPU"}


def slugify(title: str) -> str:
    return "-".join(title.lower().split())


def build_bundle() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for root, suffixes in (("sparse_gt", (".py",)), ("codes", (".alist", ".json", ".txt"))):
            for path in sorted((REPO_ROOT / root).rglob("*")):
                if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts:
                    tar.add(path, arcname=str(path.relative_to(REPO_ROOT)))
    return buffer.getvalue()


def build(out_dir: Path, username: str, experiment: str, title: str, extra_args: str) -> Path:
    template = Path(__file__).with_name("run_experiment.py")
    bundle = build_bundle()
    script = (template.read_text()
              .replace('"@@BUNDLE@@"', repr(base64.b64encode(bundle).decode("ascii")), 1)
              .replace('"@@ARGS@@"', repr(extra_args), 1)
              .replace('"@@EXPERIMENT@@"', repr(experiment), 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / template.name).write_text(script)
    metadata = {
        "id": f"{username}/{slugify(title)}",
        "title": title,
        "code_file": template.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"bundle {len(bundle) / 1024:.0f} KiB -> {out_dir / template.name}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--username", default=os.environ.get("KAGGLE_USERNAME"), required=False)
    parser.add_argument("--experiment", choices=sorted(TITLES), default="step2")
    parser.add_argument("--slug-suffix", default="", help="Distinguish parallel kernels, e.g. the code.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--args", default="", help="Extra arguments for sparse_gt.experiments.step2.")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--token-file", default=None, help="Kaggle token file (selects the account).")
    parser.add_argument("--accelerator", default=None, help="e.g. NvidiaTeslaT4")
    args = parser.parse_args()
    if not args.username:
        parser.error("--username (or KAGGLE_USERNAME) is required")

    title = TITLES[args.experiment] + (f" {args.slug_suffix}" if args.slug_suffix else "")
    out_dir = Path(args.out_dir or REPO_ROOT / "results" / f"kaggle_kernel_{slugify(title)}")
    build(out_dir, args.username, args.experiment, title, args.args)
    if args.push:
        env = dict(os.environ)
        if args.token_file:
            env["KAGGLE_API_TOKEN"] = str(Path(args.token_file).expanduser())
        command = [sys.executable, "-m", "kaggle.cli", "kernels", "push", "-p", str(out_dir)]
        if args.accelerator:
            command += ["--accelerator", args.accelerator]
        subprocess.run(command, env=env, check=True)
        print(f"status: kaggle kernels status {args.username}/{slugify(title)}")


if __name__ == "__main__":
    main()
