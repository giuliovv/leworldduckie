#!/usr/bin/env python3
"""
Latent alignment diagnostic for LeWM.

Distinguishes two explanations for the per-episode clusters seen in t-SNE:
  Case 1 (bad)  — encoder memorized episode-specific features (lighting, start pose).
                  Cross-episode frames at the same lane position are FAR apart.
  Case 2 (fine) — encoder learned lane position; episodes cluster because each covers
                  a different track region. Latent-close cross-episode pairs are also
                  pixel-similar.

Tests run:
  T1  Centroid separation vs within-episode spread  (latent index only)
  T2  Pixel–latent correlation for cross-episode pairs  (needs HDF5)
  T3  t-SNE coloured by episode and by normalised step  (latent index only)

Usage:
  python src/diagnostic_latent.py \\
      --latent-index s3://leworldduckie/evals/latent_index.npz \\
      --data-path    s3://leworldduckie/data/duckietown_100k.h5 \\
      --s3-output    s3://leworldduckie/diagnostics/

  # local latent index only (skip pixel test):
  python src/diagnostic_latent.py \\
      --latent-index /tmp/latent_index.npz

Interpretation printed to stdout and saved in diagnostic_report.txt.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _s3_download(s3_uri: str, local: str) -> str:
    import boto3
    u = urlparse(s3_uri)
    print(f'Downloading {s3_uri} …')
    boto3.client('s3', region_name='us-east-1').download_file(
        u.netloc, u.path.lstrip('/'), local)
    print(f'  → {local}  ({os.path.getsize(local)/1e6:.1f} MB)')
    return local


def _s3_upload(local: str, s3_uri: str):
    import boto3
    u = urlparse(s3_uri)
    boto3.client('s3', region_name='us-east-1').upload_file(local, u.netloc, u.path.lstrip('/'))
    print(f'Uploaded {local} → {s3_uri}')


def resolve_local(path: str, tmp_name: str) -> str:
    if path.startswith('s3://'):
        local = f'/tmp/{tmp_name}'
        if not os.path.exists(local):
            _s3_download(path, local)
        return local
    return path


# ── Load latent index ─────────────────────────────────────────────────────────

def load_index(path: str):
    local = resolve_local(path, 'latent_index.npz')
    d = np.load(local)
    all_z    = d['all_z'].astype(np.float32)   # (N, D)
    ep_idx   = d['ep_idx'].astype(np.int32)    # (N,)
    step_idx = d['step_idx'].astype(np.int32)  # (N,)
    print(f'Latent index: {all_z.shape[0]:,} frames, {all_z.shape[1]} dims, '
          f'{len(np.unique(ep_idx))} episodes')
    return all_z, ep_idx, step_idx


# ── T1: Centroid separation ───────────────────────────────────────────────────

def test_centroid_separation(all_z, ep_idx, n_episodes=40, rng=None):
    """
    Compare per-episode centroid spread to within-episode spread.

    If  centroid_spread >> within_spread  →  episodes in separate clusters (bad or just different regions).
    The ratio centroid_spread / within_spread is the key number.

    A ratio near 1 means episodes overlap in latent space.
    A ratio >> 1 means they are well separated.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    episodes = np.unique(ep_idx)
    if len(episodes) > n_episodes:
        episodes = rng.choice(episodes, n_episodes, replace=False)
        episodes.sort()

    centroids = []
    within_dists = []
    ep_sizes = []

    for ep in episodes:
        mask = ep_idx == ep
        z_ep = all_z[mask]
        if len(z_ep) < 4:
            continue
        c = z_ep.mean(axis=0)
        centroids.append(c)
        ep_sizes.append(len(z_ep))
        dists = np.linalg.norm(z_ep - c, axis=1)
        within_dists.append(dists.mean())

    centroids = np.array(centroids)
    within_mean = float(np.mean(within_dists))
    within_std  = float(np.std(within_dists))

    # pairwise centroid distances
    diff = centroids[:, None, :] - centroids[None, :, :]  # (E, E, D)
    cdist = np.linalg.norm(diff, axis=-1)                 # (E, E)
    triu = cdist[np.triu_indices(len(centroids), k=1)]
    centroid_mean = float(triu.mean())
    centroid_std  = float(triu.std())
    ratio = centroid_mean / (within_mean + 1e-8)

    lines = [
        '── T1: Centroid Separation ──────────────────────────────────',
        f'  Episodes analysed    : {len(centroids)}',
        f'  Frames/episode (med) : {np.median(ep_sizes):.0f}',
        f'  Within-episode spread: {within_mean:.3f} ± {within_std:.3f}  (mean ‖z - centroid‖)',
        f'  Centroid-to-centroid : {centroid_mean:.3f} ± {centroid_std:.3f}  (mean pairwise ‖c_i - c_j‖)',
        f'  Separation ratio     : {ratio:.2f}',
        '',
        '  Interpretation:',
    ]
    if ratio < 1.5:
        lines.append('  → ratio < 1.5  Episodes OVERLAP in latent space. Case 2 likely.')
    elif ratio < 3.0:
        lines.append('  → ratio 1.5–3  Moderate separation. Ambiguous — run T2.')
    else:
        lines.append(f'  → ratio {ratio:.1f}  Episodes are well-separated clusters.')
        lines.append('    Could be Case 1 (bad) OR Case 2 (different track regions).')
        lines.append('    Run T2 (pixel–latent correlation) to distinguish.')

    return '\n'.join(lines), {
        'centroids': centroids,
        'within_mean': within_mean,
        'centroid_mean': centroid_mean,
        'ratio': ratio,
        'episodes': episodes,
    }


# ── T2: Pixel–latent correlation ──────────────────────────────────────────────

def test_pixel_latent_correlation(all_z, ep_idx, step_idx,
                                  hdf5_path: str,
                                  n_pairs=400, rng=None):
    """
    For cross-episode frame pairs, check whether pixel-similar pairs are also
    latent-close. A positive correlation supports Case 2 (encoder tracks appearance).
    No correlation (or inverted) supports Case 1 (encoder tracks episode identity).
    """
    import h5py
    if rng is None:
        rng = np.random.default_rng(42)

    local_h5 = resolve_local(hdf5_path, 'duckietown_100k.h5')

    episodes = np.unique(ep_idx)
    n_eps = len(episodes)

    # Sample n_pairs random cross-episode pairs (i, j) with ep[i] != ep[j]
    pair_i, pair_j = [], []
    N = len(all_z)
    attempts = 0
    while len(pair_i) < n_pairs and attempts < n_pairs * 20:
        a = rng.integers(N)
        b = rng.integers(N)
        if ep_idx[a] != ep_idx[b]:
            pair_i.append(a)
            pair_j.append(b)
        attempts += 1

    pair_i = np.array(pair_i)
    pair_j = np.array(pair_j)
    all_frame_idx = np.unique(np.concatenate([pair_i, pair_j]))

    # Load pixels for needed frames
    print(f'T2: loading {len(all_frame_idx)} frames from HDF5 …')
    with h5py.File(local_h5, 'r') as f:
        # Build mapping from global frame idx → HDF5 row
        # The latent index was built on a subset (lag+frameskip filter), but
        # the HDF5 row IS the global index for ep_idx/step_idx stored in HDF5.
        # We need to find HDF5 row that matches (ep, step) for each index in all_z.
        ep_h5   = f['episode_idx'][:]
        step_h5 = f['step_idx'][:]

        # Build lookup: (ep, step) → hdf5_row
        lookup = {}
        for row, (ep, st) in enumerate(zip(ep_h5, step_h5)):
            lookup[(int(ep), int(st))] = row

        hdf5_rows = []
        valid_pairs = []
        for k, gi in enumerate(all_frame_idx):
            key = (int(ep_idx[gi]), int(step_idx[gi]))
            row = lookup.get(key)
            if row is not None:
                hdf5_rows.append(row)
                valid_pairs.append(gi)

        if not hdf5_rows:
            return 'T2: could not match latent index to HDF5 rows — skipped.', {}

        hdf5_rows = np.array(hdf5_rows)
        valid_pairs = np.array(valid_pairs)
        sorted_order = np.argsort(hdf5_rows)
        hdf5_rows_sorted = hdf5_rows[sorted_order]
        valid_pairs_sorted = valid_pairs[sorted_order]

        # Load pixels in sorted order for efficiency
        pixels_raw = f['pixels'][hdf5_rows_sorted]  # (K, 120, 160, 3) uint8

    # Map global index → loaded pixel index
    gi_to_pix = {int(gi): k for k, gi in enumerate(valid_pairs_sorted)}

    # Downsample pixels to 32×32 for fast distance computation
    from PIL import Image as PILImage
    small = np.zeros((len(valid_pairs_sorted), 32 * 32 * 3), dtype=np.float32)
    for k, px in enumerate(pixels_raw):
        img = PILImage.fromarray(px).resize((32, 32), PILImage.BILINEAR)
        small[k] = np.array(img).reshape(-1).astype(np.float32) / 255.0

    # Compute pixel and latent distances for each pair
    pixel_dists, latent_dists = [], []
    for a, b in zip(pair_i, pair_j):
        if int(a) not in gi_to_pix or int(b) not in gi_to_pix:
            continue
        ka, kb = gi_to_pix[int(a)], gi_to_pix[int(b)]
        pd = float(np.linalg.norm(small[ka] - small[kb]))
        ld = float(np.linalg.norm(all_z[a] - all_z[b]))
        pixel_dists.append(pd)
        latent_dists.append(ld)

    pixel_dists  = np.array(pixel_dists)
    latent_dists = np.array(latent_dists)

    # Spearman correlation (rank-based, robust to outliers)
    from scipy.stats import spearmanr
    corr, pval = spearmanr(pixel_dists, latent_dists)

    # Quartile analysis: bottom 25% pixel distance (most similar) vs top 25%
    q25 = np.percentile(pixel_dists, 25)
    q75 = np.percentile(pixel_dists, 75)
    lat_pixsim  = latent_dists[pixel_dists <= q25].mean()
    lat_pixdiff = latent_dists[pixel_dists >= q75].mean()

    lines = [
        '── T2: Pixel–Latent Correlation ─────────────────────────────',
        f'  Cross-episode pairs  : {len(pixel_dists)}',
        f'  Spearman ρ(pixel_d, latent_d) = {corr:.3f}  (p={pval:.3g})',
        f'  Latent dist — pixel-similar Q1 : {lat_pixsim:.3f}',
        f'  Latent dist — pixel-dissimilar Q3: {lat_pixdiff:.3f}',
        '',
        '  Interpretation:',
    ]

    if corr > 0.3 and lat_pixsim < lat_pixdiff:
        lines.append('  → ρ > 0.3 and pixel-similar pairs are latent-close.')
        lines.append('    Encoder tracks visual appearance across episodes. Case 2 (fine).')
    elif corr < 0.1:
        lines.append('  → ρ ≈ 0  No correlation between pixel and latent distances.')
        lines.append('    Encoder may be tracking episode identity, not appearance. Case 1 risk.')
    else:
        lines.append(f'  → ρ = {corr:.2f}  Weak positive correlation. Borderline.')

    return '\n'.join(lines), {
        'pixel_dists': pixel_dists,
        'latent_dists': latent_dists,
        'corr': corr,
        'pval': pval,
    }


# ── T3: t-SNE visualisation ───────────────────────────────────────────────────

def plot_tsne(all_z, ep_idx, step_idx, out_dir: str, n_sample=2000, rng=None):
    from sklearn.manifold import TSNE

    if rng is None:
        rng = np.random.default_rng(42)

    N = len(all_z)
    idx = rng.choice(N, min(n_sample, N), replace=False)
    z_s   = all_z[idx]
    ep_s  = ep_idx[idx]
    st_s  = step_idx[idx]

    print(f'T3: running t-SNE on {len(z_s)} frames …')
    t0 = time.time()
    emb = TSNE(n_components=2, perplexity=40, random_state=42,
               max_iter=1000).fit_transform(z_s)
    print(f'  done in {time.time()-t0:.1f}s')

    # Normalise step within episode
    norm_step = np.zeros(len(st_s))
    for ep in np.unique(ep_s):
        m = ep_s == ep
        mn, mx = st_s[m].min(), st_s[m].max()
        norm_step[m] = (st_s[m] - mn) / max(mx - mn, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc0 = axes[0].scatter(emb[:, 0], emb[:, 1], c=ep_s, s=4, alpha=0.6,
                          cmap='tab20')
    axes[0].set_title('t-SNE coloured by episode')
    axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')
    plt.colorbar(sc0, ax=axes[0], label='episode')

    sc1 = axes[1].scatter(emb[:, 0], emb[:, 1], c=norm_step, s=4, alpha=0.6,
                          cmap='plasma')
    axes[1].set_title('t-SNE coloured by normalised step (0=start, 1=end)')
    axes[1].set_xlabel('t-SNE 1'); axes[1].set_ylabel('t-SNE 2')
    plt.colorbar(sc1, ax=axes[1], label='norm. step')

    fig.tight_layout()
    out = os.path.join(out_dir, 'tsne.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'  Saved {out}')
    return out


def plot_pixel_latent(data: dict, out_dir: str) -> str:
    pixel_dists  = data['pixel_dists']
    latent_dists = data['latent_dists']
    corr = data['corr']

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pixel_dists, latent_dists, s=4, alpha=0.4, color='steelblue')
    ax.set_xlabel('Pixel distance (32×32 downsampled)')
    ax.set_ylabel('Latent distance (‖z_a − z_b‖)')
    ax.set_title(f'Cross-episode pairs — Spearman ρ = {corr:.3f}')
    fig.tight_layout()
    out = os.path.join(out_dir, 'pixel_latent_corr.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'  Saved {out}')
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='LeWM latent alignment diagnostic')
    ap.add_argument('--latent-index', default='s3://leworldduckie/evals/latent_index.npz')
    ap.add_argument('--data-path',    default=None,
                    help='HDF5 path for T2 pixel test (local or s3://…). Skipped if omitted.')
    ap.add_argument('--n-episodes',   type=int, default=40)
    ap.add_argument('--n-tsne',       type=int, default=2000)
    ap.add_argument('--n-pairs',      type=int, default=400,
                    help='Cross-episode pairs for T2')
    ap.add_argument('--output-dir',   default='/tmp/diag')
    ap.add_argument('--s3-output',    default=None,
                    help='s3://bucket/prefix/ to upload results')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    # Load index
    all_z, ep_idx, step_idx = load_index(args.latent_index)

    report_lines = [
        '═' * 62,
        'LeWM Latent Alignment Diagnostic',
        f'Latent index: {args.latent_index}',
        f'Frames: {len(all_z):,}   Episodes: {len(np.unique(ep_idx))}',
        '═' * 62,
        '',
    ]

    # T1
    t1_text, t1_data = test_centroid_separation(
        all_z, ep_idx, n_episodes=args.n_episodes, rng=rng)
    print(t1_text)
    report_lines += [t1_text, '']

    # T2 (optional)
    t2_plot = None
    if args.data_path:
        try:
            t2_text, t2_data = test_pixel_latent_correlation(
                all_z, ep_idx, step_idx,
                hdf5_path=args.data_path,
                n_pairs=args.n_pairs, rng=rng)
            print(t2_text)
            report_lines += [t2_text, '']
            if t2_data:
                t2_plot = plot_pixel_latent(t2_data, args.output_dir)
        except Exception as e:
            msg = f'T2 failed: {e}'
            print(msg)
            report_lines += [msg, '']
    else:
        report_lines += ['T2 skipped (--data-path not provided)', '']

    # T3
    try:
        tsne_plot = plot_tsne(all_z, ep_idx, step_idx,
                              args.output_dir, n_sample=args.n_tsne, rng=rng)
    except Exception as e:
        print(f'T3 (t-SNE) failed: {e}')
        tsne_plot = None

    report_lines += ['Output files:']
    if tsne_plot:
        report_lines.append(f'  {tsne_plot}')
    if t2_plot:
        report_lines.append(f'  {t2_plot}')

    report_text = '\n'.join(report_lines)
    report_path = os.path.join(args.output_dir, 'diagnostic_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f'\nReport saved → {report_path}')

    # Upload to S3
    if args.s3_output:
        prefix = args.s3_output.rstrip('/') + '/'
        for fname in ['diagnostic_report.txt', 'tsne.png', 'pixel_latent_corr.png']:
            local = os.path.join(args.output_dir, fname)
            if os.path.exists(local):
                _s3_upload(local, prefix + fname)

    print('\nDone.')


if __name__ == '__main__':
    main()
