"""Combine the rankings of one tuning round over all codes and pick the winner.

Reads every log.txt under results/kaggle_<round>*/ (the ranking block and the
BLER lines that run_tuning.py logs; robust to two runs sharing a log) and
writes results/<round>_RANKING.md + <round>_ranking.json. The overall score of
a config is the geometric mean over codes of its BLER / BLER(BP(50)) score.

Usage: python merge_tuning.py --round tune1 [--round tune2 ...]
"""

import argparse
import glob
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ROW = re.compile(r"^\s+([0-9.]+|nan)\s+(\S+)\s+hard-set FER (\S+)\s+(\{.*\})\s*$")
BLER = re.compile(r"^\s+(SGT \S+|BP \(50 it\))\s+((?:[0-9.]+dB:\S+\s*)+)$")


def parse_log(path):
    code = os.path.basename(os.path.dirname(path)).split("_", 1)[1]
    rows, bler = {}, {}
    for line in open(path):
        m = ROW.match(line)
        if m:
            rows.setdefault(m[2], dict(config=m[2], score=float(m[1]),
                                       hard=None if m[3] == "None" else float(m[3]), overrides=m[4]))
        m = BLER.match(line)
        if m:
            name = m[1].replace("SGT ", "")
            bler.setdefault(name, {float(e): float(b) for e, b in re.findall(r"([0-9.]+)dB:(\S+)", m[2])})
    return code, rows, bler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", action="append", required=True, help="run tag prefix, e.g. tune1")
    args = ap.parse_args()
    table, blers = {}, {}
    for r in args.round:
        for p in sorted(glob.glob(os.path.join(RES, f"kaggle_{r}*", "**", "log.txt"), recursive=True)):
            code, rows, bler = parse_log(p)
            for name, row in rows.items():
                table.setdefault(name, {}).setdefault(code, row)
            for name, b in bler.items():
                blers.setdefault(code, {}).setdefault(name, b)
    codes = sorted({c for v in table.values() for c in v})
    # overall over the codes every config was run on; the others are extra columns
    core = [c for c in codes if all(c in per for per in table.values())]
    codes = core + [c for c in codes if c not in core]
    overall = []
    for name, per in table.items():
        s = [per[c]["score"] for c in core if per[c]["score"] == per[c]["score"]]
        gm = math.exp(sum(map(math.log, s)) / len(s)) if s and len(s) == len(core) else float("nan")
        overall.append((gm, name))
    overall.sort(key=lambda x: (x[0] != x[0], x[0]))
    tag = "+".join(args.round)
    L = [f"# BP-SGT tuning: {tag}", "",
         "score = mean over Eb/N0 of BLER(config) / BLER(BP(50)) (same seeded frames); lower is better. "
         f"overall = geometric mean over the codes every config was run on ({', '.join(core)}). "
         "hard = best FER on the fixed set of BP(10) failures used for model selection.", "",
         "config | overrides | overall | " + " | ".join(f"{c} score (hard)" for c in codes),
         "---|---|---|" + "---|" * len(codes)]
    for gm, name in overall:
        per = table[name]
        ov = next(iter(per.values()))["overrides"]
        cells = [f"{per[c]['score']:.3f} ({per[c]['hard']:.3f})" if c in per and per[c]["hard"] is not None
                 else (f"{per[c]['score']:.3f}" if c in per else "-") for c in codes]
        L.append(f"{name} | `{ov}` | {gm:.3f} | " + " | ".join(cells))
    for c in codes:
        ebnos = sorted({e for b in blers[c].values() for e in b})
        L += ["", f"## {c}: BLER", "", "decoder | " + " | ".join(f"{e:g} dB" for e in ebnos),
              "---|" + "---|" * len(ebnos)]
        for name in ["BP (50 it)"] + [n for _, n in overall if n in blers[c]]:
            b = blers[c].get(name, {})
            L.append(f"{name} | " + " | ".join(f"{b[e]:.2e}" if e in b else "-" for e in ebnos))
    md = "\n".join(L) + "\n"
    with open(os.path.join(RES, f"{tag}_RANKING.md"), "w") as f:
        f.write(md)
    with open(os.path.join(RES, f"{tag}_ranking.json"), "w") as f:
        json.dump(dict(overall=overall, table=table, bler={c: {n: {str(e): v for e, v in b.items()}
                                                               for n, b in d.items()} for c, d in blers.items()}),
                  f, indent=1)
    print(md)


if __name__ == "__main__":
    main()
