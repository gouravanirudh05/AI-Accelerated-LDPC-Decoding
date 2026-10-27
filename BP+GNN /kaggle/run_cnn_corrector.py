"""Kaggle GPU job: train the ResNet correctors, then run the final sweep.

Built by ``build_kernel.py``, which inlines the source bundle into
``BUNDLE_B64`` below.  On Kaggle this:

1. unpacks ``ldpc_corrector`` + the ``GAT+BP`` BP code into ``/kaggle/working/src``;
2. installs ``torch_geometric`` if the image lacks it;
3. refuses to run without a GPU, so it never silently falls back to CPU;
4. trains ``resnet1d`` and ``resnet2d`` over seeds 42/43/44 with the configs'
   protocol (1500 steps, batch 256, mixed 2-6 dB);
5. sweeps BP(5)/BP(10)/BP(20) and all six checkpoints at 51,200 frames per
   SNR point, paired against BP(5) on identical frames;
6. tars everything into ``/kaggle/working/cnn_results.tar.gz``.

``LDPC_KAGGLE_DRYRUN=1`` runs the same pipeline on CPU with a few steps and
a few hundred frames, to check the bundle before spending GPU quota.
"""

from __future__ import annotations

import base64
import io
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import List

BUNDLE_B64 = "@@BUNDLE@@"

DRY_RUN = os.environ.get("LDPC_KAGGLE_DRYRUN") == "1"
WORK = Path(os.environ.get("LDPC_KAGGLE_WORK", "/kaggle/working"))
# Code and smoke runs stay out of /kaggle/working, which Kaggle saves as output.
SCRATCH = WORK if DRY_RUN else Path("/tmp")
SRC = SCRATCH / "ldpc_src"
RESULTS = WORK / "Results_Corrector"
LOGS = WORK / "logs"

MODELS = ("resnet1d", "resnet2d")
SEEDS = (42,) if DRY_RUN else (42, 43, 44)
DEVICE = "cpu" if DRY_RUN else "cuda"

TRAIN_OVERRIDES: List[str] = (
    ["train.steps=3", "train.eval_every=3", "train.log_every=1",
     "eval.ebn0=[4.0]", "eval.batch_size=128"]
    if DRY_RUN
    else []
)
EVAL_OVERRIDES: List[str] = (
    ["eval.frames=256", "eval.ebn0=[4.0]", "eval.batch_size=128",
     "eval.timing_batches=1"]
    if DRY_RUN
    else ["eval.frames=51200"]
)

ENV = dict(os.environ)
ENV["LDPC_GAT_BP_DIR"] = str(SRC / "GAT+BP")
ENV["PYTHONPATH"] = str(SRC)
# cuBLAS refuses deterministic mode without a fixed workspace.
ENV.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def run(args: List[str], log_name: str, check: bool = True) -> int:
    """Run a step, streaming its output to the notebook log and a file."""
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {' '.join(args)}", flush=True)
    start = time.time()
    with open(LOGS / f"{log_name}.log", "w") as log:
        process = subprocess.Popen(
            args,
            cwd=SRC,
            env=ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    print(f"[{log_name}] exit {code} after {time.time() - start:.0f}s", flush=True)
    if check and code != 0:
        raise SystemExit(f"{log_name} failed with exit code {code}; see {LOGS}")
    return code


def unpack() -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    payload = base64.b64decode(BUNDLE_B64)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as bundle:
        try:
            bundle.extractall(SRC, filter="data")
        except TypeError:  # Python without extraction filters; bundle is ours
            bundle.extractall(SRC)
    print(f"unpacked {len(payload) / 1024:.0f} KiB into {SRC}", flush=True)


def ensure_dependencies() -> None:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"], "pip")


def check_device() -> None:
    import torch

    print(f"torch {torch.__version__}", flush=True)
    if DRY_RUN:
        return
    if not torch.cuda.is_available():
        raise SystemExit(
            "No GPU in this session. Enable a GPU accelerator for the kernel "
            "(kernel-metadata.json enable_gpu, or push --accelerator)."
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


def train_args(model: str, name: str, seed: int, deterministic: bool) -> List[str]:
    return [
        sys.executable, "-m", "ldpc_corrector.train",
        "--config", f"ldpc_corrector/configs/{model}.yaml",
        "--set", f"name={name}", f"seed={seed}", f"device={DEVICE}",
        "torch_threads=0", f"deterministic={str(deterministic).lower()}",
        f"save_dir={RESULTS / 'cnn'}", *TRAIN_OVERRIDES,
    ]


def choose_determinism() -> bool:
    """Deterministic mode if every op used here supports it on this device.

    A few CUDA scatter kernels have no deterministic implementation in some
    torch builds; rather than fail the whole job, fall back and say so.
    """
    smoke = ["train.steps=2", "train.eval_every=2", "eval.ebn0=[4.0]",
             "eval.batch_size=64", f"save_dir={SCRATCH / 'ldpc_smoke'}"]
    for deterministic in (True, False):
        args = train_args("resnet2d", "smoke", 0, deterministic) + smoke
        if run(args, f"smoke_deterministic_{deterministic}", check=False) == 0:
            print(f"deterministic={deterministic}", flush=True)
            return deterministic
    raise SystemExit("Smoke run failed in both deterministic modes; see logs.")


def main() -> None:
    unpack()
    ensure_dependencies()
    check_device()
    deterministic = choose_determinism()

    checkpoints: List[Path] = []
    for model in MODELS:
        for seed in SEEDS:
            name = f"{model}_s{seed}"
            run(train_args(model, name, seed, deterministic), f"train_{name}")
            checkpoints.append(RESULTS / "cnn" / f"{name}.best.pt")

    reference = SRC / "comparison" / "test_results.csv"
    evaluate = [
        sys.executable, "-m", "ldpc_corrector.evaluate",
        "--config", "ldpc_corrector/configs/resnet1d.yaml",
        "--bp-iterations", "5", "10", "20",
        "--out-dir", str(RESULTS / "eval" / "cnn"),
        *(["--reference-csv", str(reference)] if reference.exists() else ["--no-reference"]),
    ]
    for checkpoint in checkpoints:
        evaluate += ["--checkpoint", str(checkpoint)]
    evaluate += ["--set", f"device={DEVICE}", "torch_threads=0",
                 f"deterministic={str(deterministic).lower()}", *EVAL_OVERRIDES]
    run(evaluate, "evaluate")

    archive = WORK / "cnn_results.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(RESULTS, arcname="Results_Corrector")
        tar.add(LOGS, arcname="kaggle_logs")
    print(f"\nwrote {archive}", flush=True)


if __name__ == "__main__":
    main()
