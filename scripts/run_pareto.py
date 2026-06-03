"""Pareto trade-off sweep (§3.5, Fig 14) for the diffuser problem.

Runs the multiscale TO at several desired contact-area (perimeter) targets and
plots the achieved dissipated power vs. achieved contact area. Reuses the
validated single-run driver ``run_to.py`` via subprocess so the optimization
path is identical.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "TOMAS")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(REPO, "notebooks", "config_diffuser.yaml"))
    ap.add_argument("--perims", default="40,50,60,70,80",
                    help="comma-separated desired contact-area targets")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "results", "pareto"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    perims = [float(p) for p in args.perims.split(",")]

    achieved_ca, achieved_pw = [], []
    for p in perims:
        tag = f"pareto_{p:g}"
        print(f"=== running diffuser, desired contact area = {p} ===", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "run_to.py"),
                        "--config", args.config, "--constraint", "PERIMETER",
                        "--desired-perim", str(p), "--out-dir", args.out_dir,
                        "--tag", tag], check=True)
        with open(os.path.join(args.out_dir, tag, "metrics.json")) as f:
            m = json.load(f)
        achieved_ca.append(m["final_contact_area"])
        achieved_pw.append(m["final_dissipated_power"])
        print(f"  -> contact_area={m['final_contact_area']:.2f} "
              f"power={m['final_dissipated_power']:.3f}")

    order = sorted(range(len(achieved_ca)), key=lambda i: achieved_ca[i])
    ca = [achieved_ca[i] for i in order]
    pw = [achieved_pw[i] for i in order]
    plt.figure(figsize=(6, 4))
    plt.plot(ca, pw, "o-")
    for x, y, p in zip(ca, pw, [perims[i] for i in order]):
        plt.annotate(f"{p:g}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("contact area"); plt.ylabel("dissipated power")
    plt.title("Pareto front: dissipated power vs contact area (Fig 14)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "fig14_pareto.png"), dpi=200)
    with open(os.path.join(args.out_dir, "pareto.json"), "w") as f:
        json.dump({"target_perims": perims, "achieved_contact_area": achieved_ca,
                   "achieved_power": achieved_pw}, f, indent=2)
    print(f"Saved Pareto results to {args.out_dir}")


if __name__ == "__main__":
    main()
