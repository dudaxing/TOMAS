"""Diagnose the P0 decoder's latent landscape: how is m organized over z, and
how much perimeter can a single cell deliver? Tests the hypothesis that our
designs pick multi-petal stars (extreme perimeter per cell) because our decoder
CAN express them, while the paper's smoother decoder caps per-cell perimeter,
forcing thick circle walls instead (cf. paper Figs 14/16b)."""
import argparse
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "TOMAS")
for _p in ("vae", "dataset", ""):
    sys.path.insert(0, os.path.join(REPO, _p))
import network, data_preprocess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--vae-dir", default=os.path.join(REPO, "vae"),
                help="directory holding vae_net.pt + nomalization.pt (beta-sweep variants)")
args = ap.parse_args()

NORM = ([data_preprocess.NomalizationType.LINEAR] * 8 +
        [data_preprocess.NomalizationType.LOG] * 2 +
        [data_preprocess.NomalizationType.LINEAR] * 2)
F = data_preprocess.VAE_Fields

p = network.VAE_Params(input_dim=12, encoder_hidden_dim=600, latent_dim=2,
                       decoder_hidden_dim=600)
vae = network.VariationalAutoencoder(vae_params=p)
vae.encoder.is_training = False
vae.load_state_dict(torch.load(os.path.join(args.vae_dir, "vae_net.pt")))
vae.eval()
n = torch.load(os.path.join(args.vae_dir, "nomalization.pt"))
mx, mn = n["max_feature"], n["min_feature"]
print(f"[diag] vae-dir = {args.vae_dir}")

G = 200
g = np.linspace(-3, 3, G)
Z1, Z2 = np.meshgrid(g, g)
Z = torch.tensor(np.stack([Z1.ravel(), Z2.ravel()], axis=1)).double()
with torch.no_grad():
    out = data_preprocess.stack_vae_output(vae.decoder(Z), mx, mn, NORM).numpy()

m = out[:, 2].reshape(G, G)
perim = out[:, F.shape_perim.value].reshape(G, G)
vf = out[:, F.shape_area.value].reshape(G, G)

print("== decoded fields over the 200x200 latent grid [-3,3]^2 ==")
for name, fld in [("m", m), ("perim", perim), ("vf", vf)]:
    q = np.percentile(fld, [5, 50, 90, 97, 99])
    print(f"  {name:>6}: p5={q[0]:.3f} p50={q[1]:.3f} p90={q[2]:.3f} "
          f"p97={q[3]:.3f} p99={q[4]:.3f} max={fld.max():.3f}")

# m-organization smoothness: |dm| per grid step (z-step = 6/199 = 0.030)
dmx = np.abs(np.diff(m, axis=0)); dmy = np.abs(np.diff(m, axis=1))
print(f"\n== m organization over z ==")
print(f"  mean |dm| per z-step(0.030): {0.5*(dmx.mean()+dmy.mean()):.4f}")
print(f"  fraction of grid steps with |dm|>0.5: {0.5*((dmx>0.5).mean()+(dmy>0.5).mean()):.4f}")
print(f"  -> m range crossed when z moves 1.0: ~{0.5*(dmx.mean()+dmy.mean())/0.030*1.0:.2f}")

# star region (m>6): how much perimeter does the decoder offer there?
star = m > 6
print(f"\n== star region (decoded m>6) ==")
print(f"  area share of latent grid: {star.mean():.1%}")
print(f"  decoded perim  : star-region mean={perim[star].mean():.3f}  "
      f"non-star mean={perim[~star].mean():.3f}  ratio={perim[star].mean()/perim[~star].mean():.2f}x")
print(f"  decoded vf     : star-region mean={vf[star].mean():.3f}  non-star={vf[~star].mean():.3f}")
pp = perim / np.maximum(vf, 1e-3)
print(f"  perim-per-vf   : star={pp[star].mean():.2f}  non-star={pp[~star].mean():.2f}")

# Per-cell contact-area concentration in our designs vs paper's thick walls
for tag in ["bifurcated_p0", "bifurcated_p4a"]:
    f = os.path.join(HERE, "..", "results", "to", tag, "design.npz")
    d = np.load(f, allow_pickle=True)
    cf = d["constraint_field"]; tot = cf.sum()
    s = np.sort(cf)[::-1]
    for frac in (0.10, 0.25):
        k = int(len(s) * frac)
        print(f"  [{tag}] top {int(frac*100)}% cells carry "
              f"{s[:k].sum()/tot:.1%} of contact area")
