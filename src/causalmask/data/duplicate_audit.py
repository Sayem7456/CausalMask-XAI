"""Duplicate detection and grouping for medical-image datasets.

This module provides:
- SHA-256 exact duplicate detection
- Perceptual hash computation (pHash) for near-duplicate candidate detection
- Normalized similarity verification for candidate pairs
- Duplicate cluster assignment for split-group construction

No patient ID parsing is assumed — duplicates are determined from image
content only. When reliable patient IDs exist, they can be cross-referenced.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_perceptual_hash(
    path: Path,
    hash_size: int = 8,
    highfreq_factor: int = 4,
) -> str:
    """Compute a perceptual hash (pHash) for an image.

    Uses DCT-based perceptual hashing with only PIL and NumPy.
    The hash is a hex string of length hash_size*hash_size/4.

    Args:
        path: Path to the image file.
        hash_size: Size of the final hash (hash_size x hash_size bits).
        highfreq_factor: Resize image to (hash_size*highfreq_factor) before DCT.

    Returns:
        Hex string of the perceptual hash.
    """
    img = Image.open(path).convert("L")
    img_size = hash_size * highfreq_factor
    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)

    pixels = np.array(img, dtype=np.float64)
    dct = _dct_2d(pixels)
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    bits = (dct_low > med).flatten()
    return _bits_to_hex(bits)


def _dct_2d(array: np.ndarray) -> np.ndarray:
    """Compute 2D Discrete Cosine Transform (Type II) using matrix multiplication."""
    n = array.shape[0]
    m = array.shape[1]
    result = _dct_1d(array, n, axis=0)
    result = _dct_1d(result, m, axis=1)
    return result


def _dct_1d(array: np.ndarray, n: int, axis: int = -1) -> np.ndarray:
    """Compute 1D DCT Type II."""
    x = np.asarray(array, dtype=np.float64)
    N = n
    k = np.arange(N, dtype=np.float64)[:, np.newaxis]
    n_vals = np.arange(N, dtype=np.float64)[np.newaxis, :]
    transform = np.cos(np.pi * k * (2 * n_vals + 1) / (2 * N))
    transform[0] *= np.sqrt(1.0 / N)
    transform[1:] *= np.sqrt(2.0 / N)
    return np.tensordot(transform, x, axes=([1], [axis]))


def _bits_to_hex(bits: np.ndarray) -> str:
    if len(bits) % 8 != 0:
        bits = np.pad(bits, (0, 8 - len(bits) % 8), "constant")
    ints = bits.reshape(-1, 8).dot(1 << np.arange(7, -1, -1, dtype=np.uint8))
    return ints.tobytes().hex()


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Compute Hamming distance between two perceptual hash hex strings."""
    min_len = min(len(hash_a), len(hash_b))
    a_int = int(hash_a[:min_len], 16)
    b_int = int(hash_b[:min_len], 16)
    xor = a_int ^ b_int
    return xor.bit_count()


def normalized_hamming_similarity(hash_a: str, hash_b: str) -> float:
    """Compute normalized similarity [0,1] from Hamming distance.

    1.0 means identical hashes; 0.0 means maximally different.
    """
    max_bits = min(len(hash_a), len(hash_b)) * 4
    if max_bits == 0:
        return 0.0
    dist = hamming_distance(hash_a, hash_b)
    return 1.0 - (dist / max_bits)


def find_exact_duplicates(
    manifest_df: pd.DataFrame,
    project_root: Path,
    sha256_col: str = "image_sha256",
    image_path_col: str = "image_path",
    sample_id_col: str = "sample_id",
) -> pd.DataFrame:
    """Find exact duplicate images by SHA-256 collision.

    Returns a DataFrame with columns:
        sha256, sample_ids (list), image_paths (list), cluster_id
    """
    df = manifest_df.copy()
    sha_groups = df.groupby(sha256_col)

    rows = []
    cluster_counter = itertools.count(1)
    for sha, group in sha_groups:
        if len(group) < 2:
            continue
        rows.append(
            {
                "sha256": sha,
                "sample_ids": group[sample_id_col].tolist(),
                "image_paths": group[image_path_col].tolist(),
                "cluster_id": f"exact_{next(cluster_counter)}",
                "cluster_size": len(group),
                "detection_method": "sha256_exact",
            }
        )
    return pd.DataFrame(rows)


def compute_phashes_for_manifest(
    manifest_df: pd.DataFrame,
    project_root: Path,
    image_path_col: str = "image_path",
    sample_id_col: str = "sample_id",
    hash_size: int = 8,
    highfreq_factor: int = 4,
    batch_callback: Optional[callable] = None,
) -> pd.DataFrame:
    """Compute perceptual hashes for all samples in a manifest.

    Returns DataFrame with sample_id and phash columns.
    """
    records = []
    total = len(manifest_df)
    for i, (_, row) in enumerate(manifest_df.iterrows()):
        img_path = project_root / row[image_path_col]
        try:
            phash = compute_perceptual_hash(img_path, hash_size, highfreq_factor)
        except Exception as e:
            logger.warning(f"pHash failed for {row[sample_id_col]}: {e}")
            phash = None
        records.append(
            {
                sample_id_col: row[sample_id_col],
                "phash": phash,
                "phash_hash_size": hash_size,
            }
        )
        if batch_callback and (i + 1) % max(1, total // 10) == 0:
            batch_callback(i + 1, total)
    return pd.DataFrame(records)


def find_near_duplicate_candidates(
    phash_df: pd.DataFrame,
    sample_id_col: str = "sample_id",
    phash_col: str = "phash",
    similarity_threshold: float = 0.75,
    min_similarity: float = 0.75,
) -> pd.DataFrame:
    """Find near-duplicate candidate pairs by pHash similarity.

    Only scans pairs with similarity >= min_similarity.
    Returns DataFrame with:
        sample_id_a, sample_id_b, phash_similarity, phash_distance
    """
    df = phash_df.dropna(subset=[phash_col]).copy()
    if len(df) < 2:
        return pd.DataFrame()

    phash_values = df[phash_col].values
    sample_ids = df[sample_id_col].values
    n = len(df)

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = normalized_hamming_similarity(phash_values[i], phash_values[j])
            if sim >= min_similarity and sim < 1.0:
                rows.append(
                    {
                        "sample_id_a": sample_ids[i],
                        "sample_id_b": sample_ids[j],
                        "phash_similarity": round(sim, 6),
                        "phash_distance": round(
                            1.0 - sim, 6
                        ),
                        "detection_stage": "candidate",
                    }
                )
    return pd.DataFrame(rows)


def compute_image_similarity(
    path_a: Path,
    path_b: Path,
    method: str = "mse_ssim",
) -> dict:
    """Compute image-level similarity metrics for a candidate pair.

    Args:
        path_a: Path to first image.
        path_b: Path to second image.
        method: Similarity method ('mse_ssim' uses normalized MSE + structural
                similarity approximation).

    Returns:
        Dict with similarity metrics.
    """
    img_a = Image.open(path_a).convert("L")
    img_b = Image.open(path_b).convert("L")

    size = (256, 256)
    img_a = img_a.resize(size, Image.Resampling.LANCZOS)
    img_b = img_b.resize(size, Image.Resampling.LANCZOS)

    arr_a = np.array(img_a, dtype=np.float64)
    arr_b = np.array(img_b, dtype=np.float64)

    mse = np.mean((arr_a - arr_b) ** 2)
    max_val = 255.0
    normalized_mse = mse / (max_val**2)

    ssim_approx = _ssim_approximate(arr_a, arr_b, data_range=max_val)
    peak_snr = 20 * np.log10(max_val / max(np.sqrt(mse), 1e-10))

    return {
        "mse": float(mse),
        "normalized_mse": float(normalized_mse),
        "ssim_approximate": float(ssim_approx),
        "peak_snr_db": float(peak_snr),
        "dimensions_match": img_a.size == img_b.size,
    }


def _ssim_approximate(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    data_range: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Approximate SSIM between two equal-sized grayscale arrays."""
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_a = np.mean(arr_a)
    mu_b = np.mean(arr_b)
    sigma_a_sq = np.var(arr_a)
    sigma_b_sq = np.var(arr_b)
    sigma_ab = np.mean((arr_a - mu_a) * (arr_b - mu_b))

    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a_sq + sigma_b_sq + c2)

    return float(numerator / max(denominator, 1e-10))


def verify_candidate_pairs(
    candidates_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    project_root: Path,
    image_path_col: str = "image_path",
    sample_id_col: str = "sample_id",
    ssim_threshold: float = 0.85,
) -> pd.DataFrame:
    """Verify near-duplicate candidate pairs with computed image similarity.

    Returns a DataFrame with verified pairs and their similarity metrics.
    """
    path_map = dict(
        zip(manifest_df[sample_id_col], manifest_df[image_path_col])
    )

    rows = []
    for _, row in candidates_df.iterrows():
        sid_a = row["sample_id_a"]
        sid_b = row["sample_id_b"]
        rel_a = path_map.get(sid_a)
        rel_b = path_map.get(sid_b)
        if not rel_a or not rel_b:
            continue

        abs_a = project_root / rel_a
        abs_b = project_root / rel_b

        try:
            sim = compute_image_similarity(abs_a, abs_b)
        except Exception as e:
            logger.warning(f"Similarity failed for {sid_a} vs {sid_b}: {e}")
            continue

        is_verified = sim["ssim_approximate"] >= ssim_threshold

        rows.append(
            {
                "sample_id_a": sid_a,
                "sample_id_b": sid_b,
                "phash_similarity": row["phash_similarity"],
                "ssim": sim["ssim_approximate"],
                "normalized_mse": sim["normalized_mse"],
                "peak_snr_db": sim["peak_snr_db"],
                "is_verified_near_duplicate": is_verified,
                "verification_method": "ssim_threshold",
            }
        )
    return pd.DataFrame(rows)


def assign_duplicate_clusters(
    exact_df: pd.DataFrame,
    verified_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    sample_id_col: str = "sample_id",
) -> pd.DataFrame:
    """Assign group_id and near_duplicate_cluster to every manifest sample.

    Strategy:
    - Exact duplicates => same cluster
    - Verified near-duplicates => same cluster (transitive closure)
    - No duplicate => cluster = sample_id (singleton)

    Returns a DataFrame with sample_id, group_id, near_duplicate_cluster.
    """
    from collections import defaultdict

    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    sample_ids = manifest_df[sample_id_col].tolist()
    for sid in sample_ids:
        parent[sid] = sid

    exact_cluster_counter = itertools.count(1)
    near_cluster_counter = itertools.count(1)
    cluster_map: dict[str, str] = {}
    exact_cluster_map: dict[str, str] = {}
    near_cluster_map: dict[str, str] = {}

    if not exact_df.empty:
        for _, row in exact_df.iterrows():
            ids = row["sample_ids"]
            for i in range(len(ids) - 1):
                union(ids[i], ids[i + 1])
            cid = f"exact_{next(exact_cluster_counter)}"
            for sid in ids:
                exact_cluster_map[sid] = cid

    if not verified_df.empty:
        for _, row in verified_df.iterrows():
            if row.get("is_verified_near_duplicate", False):
                union(row["sample_id_a"], row["sample_id_b"])

    for _, row in verified_df.iterrows():
        if row.get("is_verified_near_duplicate", False):
            sid_a, sid_b = row["sample_id_a"], row["sample_id_b"]
            root_a = find(sid_a)
            near_key = f"near_{root_a}"
            if near_key not in cluster_map:
                cid = f"near_{next(near_cluster_counter)}"
                cluster_map[near_key] = cid

    results = []
    for sid in sample_ids:
        root = find(sid)
        near_key = f"near_{root}"
        cluster = cluster_map.get(near_key, sid)
        exact_cluster = exact_cluster_map.get(sid, "")
        near_duplicate = cluster != sid

        results.append(
            {
                sample_id_col: sid,
                "group_id": cluster,
                "near_duplicate_cluster": cluster
                if near_duplicate
                else "",
                "exact_duplicate_cluster": exact_cluster,
                "is_exact_duplicate": bool(exact_cluster),
                "is_near_duplicate": near_duplicate,
                "cluster_size": sum(
                    1 for r in results if r["group_id"] == cluster
                )
                + (1 if cluster == sid else 0),
            }
        )

    result_df = pd.DataFrame(results)

    cluster_sizes = result_df.groupby("group_id").size().to_dict()
    result_df["cluster_size"] = result_df["group_id"].map(cluster_sizes)

    return result_df


def build_grouped_manifest(
    manifest_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    sample_id_col: str = "sample_id",
) -> pd.DataFrame:
    """Merge cluster assignments into the manifest.

    Returns a copy of manifest_df with group_id and duplicate columns added.
    """
    df = manifest_df.copy()
    cluster_subset = cluster_df[
        [
            sample_id_col,
            "group_id",
            "near_duplicate_cluster",
            "exact_duplicate_cluster",
            "is_exact_duplicate",
            "is_near_duplicate",
        ]
    ]
    merged = df.merge(cluster_subset, on=sample_id_col, how="left")

    merged["group_id"] = merged["group_id"].fillna(merged[sample_id_col])
    merged["near_duplicate_cluster"] = merged[
        "near_duplicate_cluster"
    ].fillna("")
    merged["exact_duplicate_cluster"] = merged[
        "exact_duplicate_cluster"
    ].fillna("")
    merged["is_exact_duplicate"] = merged["is_exact_duplicate"].fillna(False)
    merged["is_near_duplicate"] = merged["is_near_duplicate"].fillna(False)

    return merged
