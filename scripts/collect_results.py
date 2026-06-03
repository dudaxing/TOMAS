"""Collect all experiment metrics into a single summary table for the docs."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


print("== TO experiments ==")
print(f"{'tag':<22}{'constraint':<11}{'power':>10}{'contact_area':>14}{'max_vel':>9}{'sec':>8}")
for mp in sorted(glob.glob(os.path.join(RES, "to", "*", "metrics.json")) +
                 glob.glob(os.path.join(RES, "pareto", "*", "metrics.json"))):
    m = load(mp)
    if not m:
        continue
    print(f"{m['tag']:<22}{m['constraint_type']:<11}"
          f"{m['final_dissipated_power']:>10.3f}{m['final_contact_area']:>14.3f}"
          f"{m.get('max_velocity_magnitude', float('nan')):>9.3f}{m['runtime_sec']:>8.0f}")

print("\n== True-FEA validation (decoder vs re-homogenized) ==")
for vp in sorted(glob.glob(os.path.join(RES, "to", "*", "validation.json")) +
                 glob.glob(os.path.join(RES, "pareto", "*", "validation.json"))):
    v = load(vp)
    if not v:
        continue
    tag = os.path.basename(os.path.dirname(vp))
    print(f"{tag}: power decoder={v['decoder_power']:.3f} true={v['true_power']:.3f}"
          f" | contact decoder={v['decoder_contact_area']:.3f} true={v['true_contact_area']:.3f}")

pj = load(os.path.join(RES, "pareto", "pareto.json"))
if pj:
    print("\n== Pareto (Fig 14) ==")
    for ca, pw in sorted(zip(pj["achieved_contact_area"], pj["achieved_power"])):
        print(f"  contact_area={ca:.2f}  power={pw:.3f}")

ms = os.path.join(RES, "latent", "M_star.npy")
if os.path.exists(ms):
    import numpy as np
    d = np.load(ms, allow_pickle=True).item()
    print(f"\n== M* (3.1) ==\n  z={d['z']}\n  shape_params(a,b,m,n1,n2,n3,cx,cy)={d['shape_params']}")
