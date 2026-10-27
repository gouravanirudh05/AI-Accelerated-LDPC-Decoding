"""Kaggle GPU job: three follow-ups to the sparse-global corrector (pyg_sparse).

Built by ``build_kernel.py --experiment sparse``, which inlines ``BUNDLE_B64``.
Every variant starts from ``configs/pyg_sparse_global_gt.yaml`` at the
protocol of the original ``pyg_sparse`` runs (1000 steps), seeds 42/43/44:

``sparse``                  the baseline, re-run on this GPU
``sparse_gate``             learned per-bit gate (``model.use_gate``): can
                            switch its correction off, e.g. at 2 dB
``sparse_fer``              FER-facing loss (BCE + hard mining + soft block
                            error), the mix that helped GCN/GAT
``sparse_resume``           correction, then 5 more BP iterations resumed from
                            BP(5)'s messages, trained through them
``sparse_resume_gate_fer``  all three

and the baseline checkpoints swept *with* 5 resumed iterations but no
retraining. Swept at 51,200 frames per point against BP(5)/BP(10)/BP(20),
paired against BP(5) and BP(10) -- BP(10) is the bar a resumed variant
(BP(5) + correction + BP(5)) must clear at about equal cost.
"""

from __future__ import annotations

import base64
import io
import os
import queue
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

BUNDLE_B64 = "@@BUNDLE@@"

DRY_RUN = os.environ.get("LDPC_KAGGLE_DRYRUN") == "1"
WORK = Path(os.environ.get("LDPC_KAGGLE_WORK", "/kaggle/working"))
SCRATCH = WORK if DRY_RUN else Path("/tmp")
SRC = SCRATCH / "sparse_src"
RESULTS = WORK / "Results_Corrector_sparse"
LOGS = WORK / "logs"
DEVICE = "cpu" if DRY_RUN else "cuda"
START = time.time()

CONFIG = "ldpc_corrector/configs/pyg_sparse_global_gt.yaml"
SEEDS = (42,) if DRY_RUN else (42, 43, 44)
STEPS = ["train.steps=3", "train.eval_every=3", "eval.ebn0=[4.0]", "eval.batch_size=64"] \
    if DRY_RUN else ["train.steps=1000"]
EVAL = ["eval.frames=256", "eval.ebn0=[4.0]", "eval.batch_size=128", "eval.timing_batches=1"] \
    if DRY_RUN else ["eval.frames=51200"]

FER_LOSS = ["loss.bce=true", "loss.hard_mining=true", "loss.hard_mining_weight=0.02",
            "loss.block_soft_bler=true", "loss.block_soft_bler_weight=0.1"]
VARIANTS: List[Tuple[str, List[str]]] = [
    ("sparse", []),
    ("sparse_gate", ["model.use_gate=true"]),
    ("sparse_fer", FER_LOSS),
    ("sparse_resume", ["bp.resume_iterations=5"]),
    ("sparse_resume_gate_fer", ["bp.resume_iterations=5", "model.use_gate=true", *FER_LOSS]),
]

ENV = dict(os.environ)
ENV["LDPC_GAT_BP_DIR"] = str(SRC / "GAT+BP")
ENV["PYTHONPATH"] = str(SRC)
ENV.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
_PRINT = threading.Lock()


def log(message: str) -> None:
    with _PRINT:
        print(f"[{(time.time() - START) / 60:6.1f} min] {message}", flush=True)


def run(args: List[str], log_name: str, gpu: int = 0) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(ENV)
    if not DRY_RUN:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.time()
    with open(LOGS / f"{log_name}.log", "w") as handle:
        code = subprocess.run(args, cwd=SRC, env=env, stdout=handle,
                              stderr=subprocess.STDOUT).returncode
    log(f"{log_name}: exit {code} after {time.time() - started:.0f}s (gpu {gpu})")
    return code


def unpack() -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE_B64)), mode="r:gz") as tar:
        try:
            tar.extractall(SRC, filter="data")
        except TypeError:  # Python without extraction filters; bundle is ours
            tar.extractall(SRC)


def gpu_count() -> int:
    if DRY_RUN:
        return 1
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No GPU in this session; enable a GPU accelerator for the kernel.")
    log(f"{torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
    return torch.cuda.device_count()


def train_args(name: str, overrides: List[str], seed: int, deterministic: bool,
               save_dir: Path, extra: List[str]) -> List[str]:
    return [sys.executable, "-m", "ldpc_corrector.train", "--config", CONFIG, "--set",
            f"name={name}", f"seed={seed}", f"device={DEVICE}", "torch_threads=0",
            f"deterministic={str(deterministic).lower()}", f"save_dir={save_dir}",
            *overrides, *extra]


def evaluate_args(out_dir: Path, checkpoints: List[Path], resume: List[Path],
                  deterministic: bool, frames: List[str]) -> List[str]:
    args = [sys.executable, "-m", "ldpc_corrector.evaluate", "--config", CONFIG,
            "--bp-iterations", "5", "10", "20", "--paired-against", "BP(10)",
            "--resume-iterations", "5", "--no-reference", "--out-dir", str(out_dir)]
    for path in checkpoints:
        args += ["--checkpoint", str(path)]
    for path in resume:
        args += ["--resume-checkpoint", str(path)]
    return args + ["--set", f"device={DEVICE}", "torch_threads=0",
                   f"deterministic={str(deterministic).lower()}", *frames]


def smoke(deterministic: bool) -> bool:
    """Every variant for 2 steps, then a tiny sweep, on the real device."""
    smoke_dir = SCRATCH / "sparse_smoke"
    tiny = ["train.steps=2", "train.eval_every=2", "eval.ebn0=[4.0]", "eval.batch_size=64"]
    for name, overrides in VARIANTS:
        args = train_args(f"smoke_{name}", overrides, 0, deterministic, smoke_dir, tiny)
        if run(args, f"smoke_{name}_det{int(deterministic)}") != 0:
            return False
    checkpoints = sorted(smoke_dir.glob("*.best.pt"))
    resume = [p for p in checkpoints if p.name.startswith("smoke_sparse.")]
    sweep = evaluate_args(smoke_dir / "eval", checkpoints, resume, deterministic,
                          ["eval.frames=128", "eval.ebn0=[4.0]", "eval.batch_size=64",
                           "eval.timing_batches=1"])
    return run(sweep, f"smoke_evaluate_det{int(deterministic)}") == 0


def main() -> None:
    unpack()
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"],
                       check=True)
    gpus = gpu_count()
    for deterministic in (True, False):
        if smoke(deterministic):
            log(f"smoke passed, deterministic={deterministic}")
            break
    else:
        raise SystemExit("smoke failed in both deterministic modes; see logs/smoke_*")

    jobs = [(f"{name}_s{seed}", train_args(f"{name}_s{seed}", overrides, seed, deterministic,
                                           RESULTS, STEPS))
            for name, overrides in VARIANTS for seed in SEEDS]
    log(f"{len(jobs)} training runs over {gpus} GPU(s)")
    free: "queue.Queue[int]" = queue.Queue()
    for gpu in range(gpus):
        free.put(gpu)

    def on_free_gpu(job: Tuple[str, List[str]]) -> Tuple[str, int]:
        name, args = job
        gpu = free.get()
        try:
            return name, run(args, f"train_{name}", gpu)
        finally:
            free.put(gpu)

    with ThreadPoolExecutor(max_workers=gpus) as pool:
        outcomes = list(pool.map(on_free_gpu, jobs))
    failed = [name for name, code in outcomes if code != 0]
    if failed:
        log(f"FAILED runs: {failed}")

    checkpoints = sorted(RESULTS.glob("*.best.pt"))
    baselines = [p for p in checkpoints if p.name.startswith("sparse_s")]
    eval_code = run(evaluate_args(RESULTS / "eval", checkpoints, baselines, deterministic, EVAL),
                    "evaluate")

    archive = WORK / "sparse_results.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(RESULTS, arcname="Results_Corrector_sparse")
        tar.add(LOGS, arcname="kaggle_logs")
    log(f"wrote {archive}")
    if failed or eval_code:
        raise SystemExit(f"failed runs {failed}, evaluate exit {eval_code}")


if __name__ == "__main__":
    main()
