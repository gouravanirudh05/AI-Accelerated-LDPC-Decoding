"""End-to-end experiment: train GNN / BP-GNN decoders, BER/BLER sweep vs. BP baselines, timing + power."""

import copy
import json
import os
import time

import torch

from .bp_gnn_decoder import BPGNNDecoder
from .evaluate import base_decoders, make_decoders, model_label, osd_postprocess, simulate
from .gnn_decoder import GNNDecoder
from .neural_bp import NeuralBPDecoder
from .profiling import GPUPowerMeter, profile_decoders
from .train import gnn_loss, train_gnn


class Logger:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def profiling_decoders(code, models, bp_iters, perturb, osd_orders=(0,), osd_learned_orders=(0,),
                       bp_osd_iters=None):
    """Decoders as plain fn(llr) for timing: fixed-iteration (eager and CUDA graph),
    early-stopping, and early-stopping + OSD post-processing."""
    g = code.graph
    bp_osd_iters = bp_osd_iters or max(bp_iters)
    decs = {}
    for name, (fn, learned) in base_decoders(g, models, bp_iters, perturb).items():
        fixed = (lambda fn: lambda llr: fn(llr, False)[0])(fn)
        decs[f"{name} [fixed]"] = fixed
        if "perturb" not in name:
            decs[f"{name} [fixed, CUDA graph]"] = fixed
        es = (lambda fn: lambda llr: fn(llr, True))(fn)
        decs[f"{name} [early-stop]"] = (lambda es: lambda llr: es(llr)[0])(es)
        orders = osd_learned_orders if learned else (osd_orders if name == f"BP ({bp_osd_iters} it)" else ())
        for o in orders:
            decs[f"{name} + OSD-{o} [early-stop]"] = (lambda f: lambda llr: f(llr)[0])(
                osd_postprocess(es, g, code.rank, o))
    return decs


def batch_cap(code, hidden, budget=4e8):
    """Largest power-of-two batch whose per-edge hidden tensor stays under budget elements."""
    cap = max(1, int(budget // (code.graph.num_edges * hidden)))
    return 1 << (cap.bit_length() - 1)


MODEL_CLASSES = {"gnn": GNNDecoder, "bpgnn": BPGNNDecoder, "nbp": NeuralBPDecoder}


def _peak_mb(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 20


def measure_memory(model, code, ebno, train_batch=256, infer_batch=4096, max_batch=1 << 16):
    """Peak GPU memory of one training step and of inference, and the largest
    power-of-two training batch that fits (doubling until CUDA OOM)."""
    g = code.graph

    def train_step(b):
        c, llr = code.sample(b, ebno)
        loss = gnn_loss(model(llr, g), c, g, 0.1)[0]
        loss.backward()
        model.zero_grad(set_to_none=True)

    def infer(b):
        _, llr = code.sample(b, ebno)
        model.decode(llr, g, early_stop=False)

    model.train()
    rec = dict(train_batch=train_batch, train_peak_mb=_peak_mb(lambda: train_step(train_batch)),
               infer_batch=infer_batch)
    model.eval()
    rec["infer_peak_mb"] = _peak_mb(lambda: infer(infer_batch))
    b, ok = train_batch, train_batch
    model.train()
    while b <= max_batch:
        try:
            train_step(b)
            ok = b
            b *= 2
        except torch.cuda.OutOfMemoryError:
            break
        finally:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    model.eval()
    rec["max_train_batch"] = ok
    rec["max_train_batch_capped"] = ok >= max_batch
    rec["gpu_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    return rec


def run_experiment(exp_name, train_codes, eval_specs, model_specs, eval_cfg, prof_cfg, outdir,
                   device, log=None):
    """model_specs: list of dict(kind="gnn"|"bpgnn", kwargs=..., train=train_cfg).
    eval_specs: list of dict(code=Code, ebno=[...], prof_ebno=float, max_frames=int)."""
    os.makedirs(outdir, exist_ok=True)
    log = log or Logger(os.path.join(outdir, "log.txt"))
    models, info = [], {}
    for ms in model_specs:
        torch.manual_seed(0)
        model = MODEL_CLASSES[ms["kind"]](**ms["kwargs"]).to(device)
        model.label = ms.get("label")
        model.osd_post = ms.get("osd", True)
        label = model_label(model)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"\n=== {exp_name}: training {label} ({n_params} parameters) on "
            f"{[c.name for c in train_codes]} ===")
        t0 = time.time()
        history = train_gnn(model, train_codes, ms["train"], log)
        info[label] = dict(kind=ms["kind"], params=n_params, kwargs=ms["kwargs"], train_cfg=ms["train"],
                           train_minutes=round((time.time() - t0) / 60, 2), history=history,
                           train_codes=[c.name for c in train_codes])
        model.trained_on = [c.name for c in train_codes]
        tag = ms.get("tag", ms["kind"])
        torch.save(dict(state_dict=model.state_dict(), kind=ms["kind"], kwargs=ms["kwargs"]),
                   os.path.join(outdir, f"{exp_name}_{tag}.pt"))
        models.append(model)
        # inference-only variants of the trained model, e.g. more BP iterations per stage
        for v in ms.get("inference_variants", []):
            m2 = copy.deepcopy(model)
            for k, val in v.items():
                if k != "label":
                    setattr(m2, k, val)
            if hasattr(m2, "stages"):
                m2.iters = (m2.stages + 1) * m2.bp_iters
            m2.label = v["label"]
            info[v["label"]] = dict(info[label], variant_of=label, overrides={k: x for k, x in v.items()
                                                                             if k != "label"})
            models.append(m2)

    meter = GPUPowerMeter(0) if device.type == "cuda" else None
    if meter is not None:
        log(f"GPU: {meter.name} | NVML: {meter.available} | energy counter: {meter.has_energy}")
    results = dict(experiment=exp_name, device=str(device),
                   gpu=meter.name if meter is not None else None, models=info, codes={})
    hidden = max((ms["kwargs"].get("hidden", 1) for ms in model_specs), default=1)
    perturb = eval_cfg.get("perturb")
    if prof_cfg.get("memory") and device.type == "cuda":
        results["memory"] = {}
        for m in models:
            lab = model_label(m)
            results["memory"][lab] = r = measure_memory(m, train_codes[0], **prof_cfg["memory"])
            log(f"  memory {lab:34s} train step B={r['train_batch']}: {r['train_peak_mb']:8.1f} MB | "
                f"inference B={r['infer_batch']}: {r['infer_peak_mb']:8.1f} MB | "
                f"max train batch {r['max_train_batch']}{'+' if r['max_train_batch_capped'] else ''}")
    for spec in eval_specs:
        code = spec["code"]
        log(f"\n--- {exp_name}: evaluating on {code.name} {code.summary()} ---")
        rec = dict(code=code.summary(), ber_bler={}, macs={})
        all_models = models
        models = [m for m in all_models if getattr(m, "compatible", lambda c: True)(code)]
        for m in models:
            lab = model_label(m)
            rec["macs"][lab] = (m.macs_per_refinement(code.graph) if hasattr(m, "stages")
                                else m.macs_per_iteration(code.graph))
        cap = batch_cap(code, hidden)
        osd_kw = dict(osd_orders=tuple(eval_cfg.get("osd_orders", (0,))),
                      osd_learned_orders=tuple(eval_cfg.get("osd_learned_orders",
                                                            eval_cfg.get("osd_orders", (0,))
                                                            if eval_cfg.get("osd_learned", True) else ())),
                      bp_osd_iters=eval_cfg.get("bp_osd_iters"))
        decs = make_decoders(code, models, eval_cfg["bp_iters"], perturb, osd=eval_cfg["osd"], **osd_kw)
        for name, fn in decs.items():
            max_frames = spec.get("max_frames", eval_cfg["max_frames"])
            if "OSD" in name:
                max_frames = min(max_frames, eval_cfg.get("osd_max_frames", max_frames))
            res = simulate(fn, code, spec["ebno"], min(eval_cfg["batch"], cap), max_frames,
                           eval_cfg["min_frame_errors"], eval_cfg.get("max_seconds_per_point"))
            rec["ber_bler"][name] = res
            log(f"  {name:34s} " + " ".join(f"{r['ebno']:.2f}dB:{r['bler']:.1e}" for r in res))

        rec["prof_ebno"] = spec["prof_ebno"]
        results["codes"][code.name] = rec
        if not spec.get("profile", True):
            models = all_models
            with open(os.path.join(outdir, f"{exp_name}_results.json"), "w") as f:
                json.dump(results, f, indent=1, default=str)
            continue
        idle_w = None
        if meter is not None and meter.available:
            time.sleep(10)  # let clocks / temperature settle before the idle baseline
            idle_w = meter.idle_w(5.0)
        rec["idle_gpu_w"] = idle_w
        log(f"  profiling at Eb/N0 = {spec['prof_ebno']} dB (idle GPU {idle_w and round(idle_w, 1)} W)")
        pdecs = profiling_decoders(code, models, eval_cfg["bp_iters"], perturb, **osd_kw)
        cpu = None
        if prof_cfg.get("cpu", True) and device.type == "cuda":
            cpu_code = copy.deepcopy(code).to(torch.device("cpu"))
            cpu_models = [copy.deepcopy(m).cpu() for m in models]
            cpu = {f"{n} [fixed]": ((lambda fn: lambda llr: fn(llr, False)[0])(fn), cpu_code)
                   for n, (fn, _) in base_decoders(cpu_code.graph, cpu_models, eval_cfg["bp_iters"]).items()}
        bss = sorted({min(b, cap) for b in prof_cfg["batch_sizes"]})
        rec["profile"] = profile_decoders(
            pdecs, code, spec["prof_ebno"], bss, min(prof_cfg["energy_batch"], cap),
            meter if (meter is not None and meter.available) else None, idle_w,
            prof_cfg["energy_seconds"], prof_cfg.get("cpu_batch_sizes", (1, 64)), cpu, log)
        models = all_models
        with open(os.path.join(outdir, f"{exp_name}_results.json"), "w") as f:
            json.dump(results, f, indent=1, default=str)
    return results
