"""Re-run the trained CLIN diffusion model on the shared evaluation cases.

Same protocol and sampler as target_period_sweep.py (single-pass replacement
inpainting, SAMPLING_STEPS=1000), but on the fixed case list in
results/eval_cases.csv and saving the reconstructed fields, so the notebooks
can score land/sea/regions/distance however they like.

Modes (each saves to results/ and resumes if interrupted):
  cases     one sample per case                          -> diffusion_single.npz
  ensemble  M samples per case for the first N per era   -> diffusion_ensemble.npz
  uncond    unconditional samples (memorisation check)   -> diffusion_uncond.npz

Usage:  python run_diffusion_eval.py cases|ensemble|uncond|bench [--batch 16]

Paths (environment variables, defaults in brackets):
  CLIN_DATA_DIR  data folder              [found by clin_common.py]
  CLIN_CODE_DIR  repo code folder         [./clin_ubelix_bundle/code]
  CLIN_CKPT      trained checkpoint (.pt) [./model/clin_era5_202609260333_best_00236.pt]
  CLIN_NORM      norm_stats.npy           [next to the checkpoint]
Cost: ~16 s per 1000-step reconstruction on a laptop RTX PRO 2000 (fp32).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import clin_common as cc  # noqa: E402  (resolves the data dir)

os.environ.setdefault("CLIN_DATA_DIR", str(cc.DATA_DIR))
os.environ["CLIN_MODEL_DIR"] = str(ROOT / "results" / "module_log")   # the module writes its log here
sys.path.insert(0, os.environ.get("CLIN_CODE_DIR", str(ROOT / "clin_ubelix_bundle" / "code")))

import torch  # noqa: E402
import clin_era5_reconstruction as cer  # noqa: E402

CKPT = Path(os.environ.get("CLIN_CKPT", ROOT / "model" / "clin_era5_202609260333_best_00236.pt"))
NORM = Path(os.environ.get("CLIN_NORM", CKPT.parent / "norm_stats.npy"))
ENS_PER_ERA, ENS_MEMBERS, N_UNCOND = 10, 8, 64


def load_model(device):
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    model = cer.CLINUNet().to(device)
    model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ckpt["model"].items()})
    model.eval()
    print(f"loaded {CKPT.name}: step {ckpt.get('global_step')}, best_loss {ckpt.get('best_loss')}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    return model


class Setup:
    def __init__(self):
        self.lat, self.lon, elev, self.lsm = cc.load_static()
        self.elev_n = (elev - elev.mean()) / (elev.std() + 1e-6)          # as in the sweep
        self.t_mean, self.t_std = np.load(NORM)
        self.sin_doy, self.cos_doy = cer.load_doy_fourier(cer.DATA_DIR)
        self.T, self.dates, self.doly = cc.load_era5(*cc.TEST_YEARS)
        H, hdates, hdoly, meta, self.cell = cc.load_hist(self.lat, self.lon)
        self.obs = np.isfinite(H)
        self.cases = cc.get_cases(hdates, hdoly, self.obs, self.dates, self.doly)

    def cond(self, date):
        pos = (date - cc.T0).days            # array position in the 1667..2025 doy files
        s = np.full(self.lsm.shape, self.sin_doy[pos], np.float32)
        c = np.full(self.lsm.shape, self.cos_doy[pos], np.float32)
        return np.stack([self.elev_n, self.lsm, s, c])

    def batch(self, rows):
        kv, km, cd = [], [], []
        for r in rows:
            mask = np.zeros(self.lsm.size, bool)
            mask[self.cell[self.obs[r.hist_row]]] = True
            mask = mask.reshape(self.lsm.shape)
            truth_n = (self.T[r.test_row] - self.t_mean) / self.t_std
            kv.append((truth_n * mask)[None]); km.append(mask[None]); cd.append(self.cond(self.dates[r.test_row]))
        f = lambda a: torch.from_numpy(np.stack(a).astype(np.float32)).cuda()
        return f(kv), f(km), f(cd)


def run_conditional(su, model, sched, rows, out_path, members, bs):
    """rows x members reconstructions; resumable via the saved 'done' flags."""
    n = len(rows)
    shape = (n, members) + su.lsm.shape
    if out_path.exists():
        z = np.load(out_path)
        recon, done = z["recon"], z["done"]
    else:
        recon, done = np.zeros(shape, np.float16), np.zeros((n, members), bool)
    jobs = [(i, m) for i in range(n) for m in range(members) if not done[i, m]]
    print(f"{len(jobs)} reconstructions to do -> {out_path.name}")
    for j0 in range(0, len(jobs), bs):
        chunk = jobs[j0:j0 + bs]
        t = time.time()
        torch.manual_seed(1000 * chunk[0][0] + chunk[0][1])
        kv, km, cd = su.batch([rows[i] for i, _ in chunk])
        x = cer.clin_reconstruct(model, sched, kv, km, cd, n_steps=cer.SAMPLING_STEPS, device="cuda")
        x = x[:, 0].cpu().numpy() * su.t_std + su.t_mean
        for k, (i, m) in enumerate(chunk):
            recon[i, m] = x[k]; done[i, m] = True
        np.savez(out_path, recon=recon, done=done, case=np.array([r.Index for r in rows]))
        print(f"  {done.sum()}/{done.size} done ({time.time()-t:.0f}s for {len(chunk)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "cases", "ensemble", "uncond"])
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    sched = cer.LambdaSchedule(device="cuda")
    model = load_model("cuda")
    su = Setup()
    rows = list(su.cases.itertuples())
    print(f"{len(rows)} cases, norm mean/std {su.t_mean:.3f}/{su.t_std:.3f}")

    if a.mode == "bench":
        kv, km, cd = su.batch(rows[:a.batch])
        lam = sched.u_to_lambda(torch.full((a.batch,), 0.5, device="cuda"))
        with torch.no_grad():
            for _ in range(3): model(kv, lam, cd)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(20): model(kv, lam, cd)
            torch.cuda.synchronize()
        dt = (time.time() - t) / 20
        print(f"batch {a.batch}: {dt*1000:.0f} ms/step -> {dt*1000/60:.1f} min per batch of 1000 steps, "
              f"peak mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    elif a.mode == "cases":
        run_conditional(su, model, sched, rows, cc.RESULTS_DIR / "diffusion_single.npz", 1, a.batch)
    elif a.mode == "ensemble":
        sub = [r for r in rows if r.Index in set(su.cases.groupby("era").head(ENS_PER_ERA).index)]
        run_conditional(su, model, sched, sub, cc.RESULTS_DIR / "diffusion_ensemble.npz", ENS_MEMBERS, a.batch)
    elif a.mode == "uncond":
        out = cc.RESULTS_DIR / "diffusion_uncond.npz"
        rng = np.random.default_rng(1)
        # random dates in the validation decade (1950-59): seasons spread evenly, fields never trained on
        dates = list(cc.pd.Timestamp("1950-01-01") + cc.pd.to_timedelta(rng.integers(0, 3652, N_UNCOND), unit="D"))
        samples = []
        for b0 in range(0, N_UNCOND, a.batch):
            torch.manual_seed(b0)
            cd = torch.from_numpy(np.stack([su.cond(d) for d in dates[b0:b0 + a.batch]])).cuda()
            x = cer.sample_unconditional(model, sched, cd, n_steps=cer.SAMPLING_STEPS, device="cuda")
            samples.append(x[:, 0].cpu().numpy() * su.t_std + su.t_mean)
            print(f"  {b0 + len(cd)}/{N_UNCOND}", flush=True)
        np.savez(out, samples=np.concatenate(samples), dates=np.array([str(d.date()) for d in dates]))


if __name__ == "__main__":
    main()
