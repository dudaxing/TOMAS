"""Pareto trade-off sweep (§3.5, Fig 14) with warm-start continuation.

Sweeps the desired contact area in ascending order and **warm-starts** each
optimization from the previous converged network (a homotopy / continuation
scheme). This yields a smooth, monotonic front instead of the noisy one obtained
when every point is optimized from scratch (different local optima).

Each design is additionally validated by true-FEA re-homogenization
(run_validate_design.py) so the front can be reported in *true* dissipated power,
not just the decoder's estimate.
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
    ap.add_argument("--no-warm-start", action="store_true",
                    help="optimize each point from scratch (old behaviour)")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip true-FEA validation of each point")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    perims = sorted(float(p) for p in args.perims.split(","))  # ascending -> continuation

    ca, dec_pw, true_pw = [], [], []
    prev_net = None
    for p in perims:
        tag = f"pareto_{p:g}"
        outdir = os.path.join(args.out_dir, tag)
        cmd = [sys.executable, os.path.join(HERE, "run_to.py"),
               "--config", args.config, "--constraint", "PERIMETER",
               "--desired-perim", str(p), "--out-dir", args.out_dir, "--tag", tag]
        if prev_net and not args.no_warm_start:
            cmd += ["--init-net", prev_net]
        print(f"=== diffuser, desired contact area = {p}"
              f"{' (warm-start)' if (prev_net and not args.no_warm_start) else ''} ===", flush=True)
        subprocess.run(cmd, check=True)
        with open(os.path.join(outdir, "metrics.json")) as f:
            m = json.load(f)
        ca.append(m["final_contact_area"]); dec_pw.append(m["final_dissipated_power"])
        prev_net = os.path.join(outdir, "net.pt")

        tp = None
        if not args.no_validate:
            subprocess.run([sys.executable, os.path.join(HERE, "run_validate_design.py"),
                            "--config", args.config,
                            "--design", os.path.join(outdir, "design.npz")], check=True)
            with open(os.path.join(outdir, "validation.json")) as f:
                tp = json.load(f)["true_power"]
        true_pw.append(tp)
        print(f"  -> contact_area={m['final_contact_area']:.2f} decoder_power="
              f"{m['final_dissipated_power']:.3f} true_power={tp}")

    # plot (sorted by achieved contact area)
    order = sorted(range(len(ca)), key=lambda i: ca[i])
    cas = [ca[i] for i in order]
    plt.figure(figsize=(6, 4))
    plt.plot(cas, [dec_pw[i] for i in order], "o-", label="decoder")
    if not args.no_validate:
        plt.plot(cas, [true_pw[i] for i in order], "s--", label="true (FEA)")
    plt.xlabel("contact area"); plt.ylabel("dissipated power")
    plt.title("Pareto front (Fig 14): warm-start continuation")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "fig14_pareto.png"), dpi=200)
    with open(os.path.join(args.out_dir, "pareto.json"), "w") as f:
        json.dump({"target_perims": perims, "achieved_contact_area": ca,
                   "achieved_power": dec_pw, "true_power": true_pw,
                   "warm_start": not args.no_warm_start}, f, indent=2)
    # monotonicity check
    pw_sorted = [(true_pw[i] if true_pw[i] is not None else dec_pw[i]) for i in order]
    mono = all(pw_sorted[i] <= pw_sorted[i + 1] + 1e-9 for i in range(len(pw_sorted) - 1))
    print(f"Saved Pareto to {args.out_dir}. Monotonic (power increases with contact area): {mono}")


if __name__ == "__main__":
    main()
