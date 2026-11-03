"""Plots and markdown tables from experiment result dicts."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"BP-GNN": "tab:blue", "GNN": "tab:green", "BP10": "tab:orange", "BP50": "tab:brown",
          "perturb": "tab:pink", "NMS": "tab:olive", "NBP": "tab:red", "BP-GNN-det": "tab:cyan",
          "NBP-shared": "tab:purple"}


def _style(name):
    if name.startswith("BP-GNN detached"):
        key, mk = "BP-GNN-det", "o"
    elif name.startswith("BP-GNN"):
        key, mk = "BP-GNN", "o"
    elif name.startswith("NBP shared"):
        key, mk = "NBP-shared", "v"
    elif name.startswith("NBP"):
        key, mk = "NBP", "v"
    elif name.startswith("GNN"):
        key, mk = "GNN", "^"
    elif "perturb" in name:
        key, mk = "perturb", "d"
    elif name.startswith("BP (10"):
        key, mk = "BP10", "s"
    elif name.startswith("BP"):
        key, mk = "BP50", "s"
    else:
        key, mk = "NMS", "x"
    ls = "-." if "OSD" in name else ("-" if ("GNN" in name or "NBP" in name) else "--")
    return dict(color=COLORS[key], linestyle=ls, marker=mk, markersize=4)


def plot_code(rec, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, res in rec["ber_bler"].items():
        x = [r["ebno"] for r in res]
        for ax, key in zip(axes, ("ber", "bler")):
            y = [r[key] if r[key] > 0 else float("nan") for r in res]
            ax.semilogy(x, y, label=name, **_style(name))
    for ax, lab in zip(axes, ("BER", "BLER")):
        ax.set_xlabel("$E_b/N_0$ [dB]")
        ax.set_ylabel(lab)
        ax.grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_training(results, path):
    models = {k: v for k, v in results["models"].items() if v["history"]}
    if not models:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for lab, info in models.items():
        h = info["history"]
        ax[0].plot([r["step"] for r in h], [r["loss"] for r in h], label=lab)
        ax[1].semilogy([r["step"] for r in h], [max(r["val_bler"], 1e-6) for r in h], label=lab)
    ax[0].set_ylabel("training loss")
    ax[1].set_ylabel("validation BLER")
    for a in ax:
        a.set_xlabel("step")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f"{results['experiment']}: training")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_profile(results, path):
    codes = [c for c, r in results["codes"].items() if r.get("profile")]
    if not codes:
        return
    names = list(results["codes"][codes[0]]["profile"])
    metrics = [("latency B=1 [ms]", lambda p: list(p["gpu"].values())[0]["median_ms"]),
               ("throughput [cw/s]", lambda p: list(p["gpu"].values())[-1]["throughput_cw_s"]),
               ("energy / codeword [uJ]", lambda p: p.get("gpu_energy", {}).get("energy_per_cw_uj", 0))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.5 * len(names) * len(codes) / 2 + 3))
    for ax, (lab, f) in zip(axes, metrics):
        w = 0.8 / len(codes)
        for i, c in enumerate(codes):
            prof = results["codes"][c]["profile"]
            vals = [f(prof[n]) if prof.get(n, {}).get("gpu") else 0 for n in names]
            ax.barh([j + i * w for j in range(len(names))], vals, height=w, label=c)
        ax.set_yticks([j + 0.4 - w / 2 for j in range(len(names))])
        ax.set_yticklabels(names if ax is axes[0] else [], fontsize=8)
        ax.set_xscale("log")
        ax.set_title(lab)
        ax.grid(alpha=0.3, axis="x")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def markdown(results):
    L = [f"## {results['experiment']}", "", f"GPU: {results.get('gpu')}", ""]
    for lab, info in results["models"].items():
        L.append(f"* {lab}: {info['params']} parameters, trained {info['train_minutes']} min on "
                 f"{', '.join(info['train_codes'])}")
    L.append("")
    if results.get("memory"):
        L += ["### GPU memory (measured on the training code)", "",
              "decoder | train step peak [MB] (batch) | max train batch | inference peak [MB] (batch)",
              "---|---|---|---"]
        for lab, r in results["memory"].items():
            mb = f"{r['max_train_batch']}{'+' if r['max_train_batch_capped'] else ''}"
            L.append(f"{lab} | {r['train_peak_mb']:.1f} ({r['train_batch']}) | {mb} | "
                     f"{r['infer_peak_mb']:.1f} ({r['infer_batch']})")
        L.append("")
    for cname, rec in results["codes"].items():
        c = rec["code"]
        L += [f"### {cname}  (n={c['n']}, k={c['k']}, R={c['rate']}, edges={c['edges']})", ""]
        names = list(rec["ber_bler"])
        ebnos = sorted({r["ebno"] for res in rec["ber_bler"].values() for r in res})
        L.append("BLER | " + " | ".join(f"{e:.1f} dB" for e in ebnos))
        L.append("---|" + "---|" * len(ebnos))
        for n in names:
            d = {r["ebno"]: r for r in rec["ber_bler"][n]}
            L.append(f"{n} | " + " | ".join(
                (f"{d[e]['bler']:.2e}" if e in d else "-") for e in ebnos))
        L += ["", "BER | " + " | ".join(f"{e:.1f} dB" for e in ebnos), "---|" + "---|" * len(ebnos)]
        for n in names:
            d = {r["ebno"]: r for r in rec["ber_bler"][n]}
            L.append(f"{n} | " + " | ".join(
                (f"{d[e]['ber']:.2e}" if e in d else "-") for e in ebnos))
        idle = rec.get("idle_gpu_w")
        if not rec.get("profile"):
            L.append("")
            continue
        L += ["", f"Inference profile at Eb/N0 = {rec['prof_ebno']} dB "
                  f"(MACs / codeword: " + ", ".join(f"{k} {v:,} per GNN call" for k, v in rec["macs"].items())
                  + "; "
                  f"idle GPU power {idle and round(idle, 1)} W)", "",
              "decoder [mode] | GPU lat. B=1 [ms] | GPU lat. large B [ms] | GPU throughput [cw/s] | "
              "avg GPU power [W] | energy/cw [uJ] | dyn. energy/cw [uJ] | CPU lat. B=1 [ms]",
              "---|---|---|---|---|---|---|---"]
        for n, p in rec["profile"].items():
            if not p.get("gpu"):
                continue
            g = list(p["gpu"].items())
            e = p.get("gpu_energy", {})
            cpu = p.get("cpu", {})
            cpu1 = cpu.get(1) or cpu.get("1")
            cpu_s = f"{cpu1['median_ms']:.2f}" if cpu1 else "-"
            nan = float("nan")
            L.append(f"{n} | {g[0][1]['median_ms']:.3f} | {g[-1][1]['median_ms']:.2f} (B={g[-1][0]}) | "
                     f"{g[-1][1]['throughput_cw_s']:.0f} | {e.get('avg_w', nan):.1f} | "
                     f"{e.get('energy_per_cw_uj', nan):.2f} | "
                     f"{e.get('dynamic_energy_per_cw_uj', nan):.2f} | {cpu_s}")
        L.append("")
    return "\n".join(L)


def write_report(results, outdir):
    os.makedirs(outdir, exist_ok=True)
    exp = results["experiment"]
    for cname, rec in results["codes"].items():
        plot_code(rec, os.path.join(outdir, f"{exp}_{cname}_ber_bler.png"), f"{cname}")
    plot_training(results, os.path.join(outdir, f"{exp}_training.png"))
    plot_profile(results, os.path.join(outdir, f"{exp}_profile.png"))
    md = markdown(results)
    with open(os.path.join(outdir, f"{exp}_summary.md"), "w") as f:
        f.write(md + "\n")
    return md
