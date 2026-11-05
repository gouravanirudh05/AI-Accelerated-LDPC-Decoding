"""Build self-contained Kaggle kernels for the BP-SGT experiments.

The sources (this project's sgt/, tuning/, final/ and the BP-GNN project's
common/, normal_ldpc/, 5g_ldpc/, final/ incl. code matrices) are embedded as a
base64 tarball with the folder layout of the repository. A kernel has one run
queue per GPU; on a T4 x2 session the queues run in parallel (CUDA_VISIBLE_DEVICES
= queue index), otherwise one after another. Results land in
/kaggle/working/results/<run tag>.

  python kaggle/build_kernel.py [--only name,...]     # writes kaggle/<name>/{script,metadata}
  KAGGLE_API_TOKEN=... kaggle kernels push -p kaggle/<name>
"""

import argparse
import base64
import io
import json
import os
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NV = "Nvidia_GNN+BP(lLatest)"
SOURCES = [(os.path.join(ROOT, d), f"SparseGT_BP/{d}") for d in ("sgt", "tuning", "final")] + \
          [(os.path.join(ROOT, "..", NV, d), f"{NV}/{d}") for d in ("common", "normal_ldpc", "5g_ldpc", "final")]
A, B, C = "gouravanirudh", "gouravanirudh1105", "gourasri"
D = "arnavoruganty"  # lent for this project only
TUNE = "SparseGT_BP/tuning/run_tuning.py"
FINAL = "SparseGT_BP/final/run_sgt_final.py"
PROFILE = "SparseGT_BP/final/profile_sgt.py"


def tune(code, configs, tag="tune1", minutes=20, grid=None):
    a = ["--code", code, "--configs", ",".join(configs), "--minutes", str(minutes), "--tag", tag]
    return TUNE, a + (["--grid", grid] if grid else [])


def final(group, codes, config="sgt_config.json"):
    return FINAL, ["--group", group, "--codes", ",".join(codes), "--config", config]


R1A = ["base", "k05", "k20", "k30", "L3", "d32"]
R1B = ["local", "unsat", "replace", "llrbase", "lr5e4"]
R1X = ["nobias", "h8", "k15", "plateau", "hard50", "syn0"]
R1L = ["base", "k05", "k20", "local", "L3", "unsat"]  # transfer of the main factors to larger codes
KERNELS = {
    "smoke": dict(title="SGT LDPC Smoke", user=A, gpu=False, queues=[[
        (TUNE, ["--quick", "--code", "mackay_204_102", "--configs", "base,local", "--tag", "smoke"]),
        (FINAL, ["--quick", "--group", "5g", "--codes", "5G_BG2_k64_n128"])]]),
    # round 1: one factor at a time around the base config, three codes
    "tune1_mackay204": dict(title="SGT LDPC Tune1 MacKay204", user=A, gpu=True, queues=[
        [tune("mackay_204_102", R1A, "tune1a")], [tune("mackay_204_102", R1B, "tune1b")]]),
    "tune1_wimax": dict(title="SGT LDPC Tune1 WiMAX", user=A, gpu=True, queues=[
        [tune("wimax_576_288", R1A, "tune1a")], [tune("wimax_576_288", R1B, "tune1b")]]),
    "tune1_5g128": dict(title="SGT LDPC Tune1 5G128", user=B, gpu=True, queues=[
        [tune("5G_BG2_k64_n128", R1A, "tune1a")], [tune("5G_BG2_k64_n128", R1B, "tune1b")]]),
    "tune1x_mackay_5g": dict(title="SGT LDPC Tune1X MacKay204 5G128", user=C, gpu=True, queues=[
        [tune("mackay_204_102", R1X, "tune1c")], [tune("5G_BG2_k64_n128", R1X, "tune1c")]]),
    "tune1x_wimax": dict(title="SGT LDPC Tune1X WiMAX", user=C, gpu=True, queues=[
        [tune("wimax_576_288", R1X[:3], "tune1c")], [tune("wimax_576_288", R1X[3:], "tune1d")]]),
    "tune1l_bg1_704": dict(title="SGT LDPC Tune1L 5G704", user=D, gpu=True, queues=[
        [tune("5G_BG1_k528_n704", R1L[:3], "tune1e")], [tune("5G_BG1_k528_n704", R1L[3:], "tune1f")]]),
    "tune1l_mackay408": dict(title="SGT LDPC Tune1L MacKay408", user=D, gpu=True, queues=[
        [tune("mackay_408_204", R1L[:3], "tune1e")], [tune("mackay_408_204", R1L[3:], "tune1f")]]),
}
# round 2 (tuning/round2.json): 10 configs x 4 codes, four GPU queues per code (two kernels per account)
R2 = [["r2_k30", "r2_k50", "r2_k30c"], ["r2_w10", "r2_w30", "r2_w50"], ["r2_w30c", "r2_w20"],
      ["r2_w30cos", "r2_w50c"]]
for user, code, short in [(A, "5G_BG2_k64_n128", "5G128"), (B, "mackay_204_102", "MacKay204"),
                          (C, "wimax_576_288", "WiMAX"), (D, "5G_BG1_k528_n704", "5G704")]:
    for half in (0, 1):
        KERNELS[f"tune2_{short.lower()}_{'ab'[half]}"] = dict(
            title=f"SGT LDPC Tune2 {short} {'AB'[half]}", user=user, gpu=True,
            queues=[[tune(code, R2[2 * half + q], f"tune2{'abcd'[2 * half + q]}", grid="round2.json")]
                    for q in (0, 1)])

# round 3 (tuning/round3.json): warm start with BP(50)-failure mining, 30 min, one kernel per code
R3 = [["r3_k50", "r3_w50m50"], ["r3_w20m50", "r3_w50m50lr"]]
for user, code, short in [(A, "5G_BG2_k64_n128", "5G128"), (B, "mackay_204_102", "MacKay204"),
                          (C, "wimax_576_288", "WiMAX"), (D, "5G_BG1_k528_n704", "5G704")]:
    KERNELS[f"tune3_{short.lower()}"] = dict(
        title=f"SGT LDPC Tune3 {short}", user=user, gpu=True,
        queues=[[tune(code, R3[q], f"tune3{'ab'[q]}", minutes=30, grid="round3.json")] for q in (0, 1)])
# final runs of the tuned model (final/sgt_config.json), 10 codes over 4 accounts x 2 GPU queues
N, G = "normal", "5g"
FINAL_SPLIT = {
    A: [[(N, ["mackay_96_48"]), (N, ["ccsds_128_64"])], [(G, ["5G_BG1_k1056_n1408"])]],
    B: [[(N, ["mackay_204_102"])], [(G, ["5G_BG2_k64_n128"]), (G, ["5G_BG2_k128_n256"])]],
    C: [[(N, ["mackay_408_204"])], [(G, ["5G_BG2_k256_n512"])]],
    D: [[(N, ["wimax_576_288"])], [(G, ["5G_BG1_k528_n704"])]],
}
for cfg, short in [("sgt_config.json", "")]:
    for i, (user, queues) in enumerate(FINAL_SPLIT.items()):
        KERNELS[f"final{short}_{'abcd'[i]}"] = dict(
            title=f"SGT LDPC Final{short.title()} {'ABCD'[i]}", user=user, gpu=True,
            queues=[[final(g, c, cfg) for g, c in q] for q in queues])

# clean re-profiling of the final checkpoints: one process on GPU 0 per account, reading the
# account's own final kernel output (kernel_sources) -- see final/profile_sgt.py
for i, (user, queues) in enumerate(FINAL_SPLIT.items()):
    codes = [c for q in queues for _, cs in q for c in cs]
    KERNELS[f"profile_{'abcd'[i]}"] = dict(
        title=f"SGT LDPC Profile {'ABCD'[i]}", user=user, gpu=True,
        sources=[f"{user}/sgt-ldpc-final-{'abcd'[i]}"],
        queues=[[(PROFILE, ["--codes", ",".join(codes)])]])


TEMPLATE = '''"""Kaggle entry point (generated by SparseGT_BP/kaggle/build_kernel.py): {title}."""

import base64
import io
import os
import shutil
import subprocess
import sys
import tarfile
import threading

BUNDLE = {bundle!r}
SRC = "/kaggle/working/src"
QUEUES = {queues!r}

with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz") as t:
    t.extractall(SRC, filter="data")
try:
    import pynvml  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nvidia-ml-py"], check=False)
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)
try:
    import torch
    ngpu = torch.cuda.device_count()
except Exception:
    ngpu = 0
rcs = [0] * len(QUEUES)


def run_queue(i):
    env = dict(os.environ)
    if ngpu > 1:
        env["CUDA_VISIBLE_DEVICES"] = str(i % ngpu)
    for script, args in QUEUES[i]:
        print(f"[queue {{i}}] {{script}} {{args}}", flush=True)
        rcs[i] |= subprocess.run([sys.executable, os.path.join(SRC, script)] + args, env=env).returncode


if ngpu > 1:
    threads = [threading.Thread(target=run_queue, args=(i,)) for i in range(len(QUEUES))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
else:
    for i in range(len(QUEUES)):
        run_queue(i)
shutil.rmtree(SRC, ignore_errors=True)
print("exit codes", rcs)
sys.exit(max(rcs))
'''


def bundle():
    buf = io.BytesIO()
    skip = lambda ti: None if ("__pycache__" in ti.name or ti.name.endswith((".pyc", ".pt"))) else ti
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for src, arc in SOURCES:
            t.add(os.path.normpath(src), arc, filter=skip)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated kernel names (default: all)")
    args = ap.parse_args()
    b = bundle()
    names = args.only.split(",") if args.only else list(KERNELS)
    for name in names:
        k = KERNELS[name]
        d = os.path.join(ROOT, "kaggle", name)
        os.makedirs(d, exist_ok=True)
        slug = k["title"].lower().replace(" ", "-")
        fname = f"{slug}.py"
        with open(os.path.join(d, fname), "w") as f:
            f.write(TEMPLATE.format(title=k["title"], bundle=b, queues=k["queues"]))
        meta = dict(id=f"{k['user']}/{slug}", title=k["title"], code_file=fname, language="python",
                    kernel_type="script", is_private=True, enable_gpu=k["gpu"], enable_tpu=False,
                    enable_internet=True, dataset_sources=[], kernel_sources=k.get("sources", []),
                    competition_sources=[], model_sources=[])
        if k["gpu"]:
            meta["machine_shape"] = "NvidiaTeslaT4"
        with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"built {d}/{fname} ({os.path.getsize(os.path.join(d, fname)) // 1024} KiB) -> {meta['id']}")


if __name__ == "__main__":
    main()
