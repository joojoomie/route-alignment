#!/usr/bin/env python3
"""Unsupervised VLAD aggregation of DINOv2 patch tokens (development variant).

The shipped descriptor is the mean of the masked patch tokens. A mean is owned
by area: road surface, foliage and sky decide it, and the small structures that
tell one place from the next (a gate, a signboard, a break in a fence) are
averaged away. That is the mechanism behind the flat similarity rows in the
self-similar corridors and behind the collapsed structure evidence at the
kinks.

VLAD keeps those structures: tokens are assigned to a small vocabulary of
visual words and the residuals are summed per word, so a word that occurs in
one small region still gets its own block of the descriptor. Following AnyLoc,
the vocabulary is learned without labels, here from the route's own tokens
(both runs of one camera, so the words are shared by the two sequences being
matched). Nothing is trained; there is no held-out number for this variant.

Pipeline: cache the masked patch tokens per frame once (float16), fit a k-means
vocabulary per camera, aggregate, intra-normalise, L2-normalise. Consumers see
an ordinary L2-normalised descriptor array through the usual cache path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import build_reliability_masks
import frame_service


ROOT = Path(__file__).resolve().parents[1]

VOCABULARY_SIZE = 32
VOCABULARY_FRAME_STRIDE = 8      # every 8th frame of each run feeds the vocabulary
KMEANS_ITERATIONS = 25
KMEANS_SEED = 0
TOKEN_DTYPE = np.float16


def token_cache_path(run: str, camera: str) -> Path:
    import build_task2_unified as unified

    digest = frame_service.file_sha256(frame_service.source_path(run, camera))
    key = frame_service.cache_key(run, camera, *frame_service.DESCRIPTOR_SIZE, digest)
    version = build_reliability_masks.mask_version(unified.VARIANT.mask_variant)
    grey = "-grey" if unified.VARIANT.greyscale else ""
    return unified.OUTPUT_DIR / camera / f"tokens_{key}_mask{version}_{unified.DESCRIPTOR_VERSION}{grey}.npy"


def vocabulary_path(camera: str) -> Path:
    import build_task2_unified as unified

    version = build_reliability_masks.mask_version(unified.VARIANT.mask_variant)
    return unified.OUTPUT_DIR / camera / f"vlad_vocabulary_k{VOCABULARY_SIZE}_mask{version}_seed{KMEANS_SEED}.npy"


def extract_tokens(run: str, camera: str, force: bool = False) -> np.ndarray:
    """Masked, L2-normalised DINOv2 patch tokens for every frame: [frames, valid_patches, dim]."""

    import build_task2_unified as unified

    target = token_cache_path(run, camera)
    if target.is_file() and not force:
        return np.load(target, mmap_mode="r")

    import torch
    from transformers import AutoImageProcessor, Dinov2Model

    torch.set_num_threads(1)
    processor = AutoImageProcessor.from_pretrained(str(unified.DINO_DIR), local_files_only=True)
    model = Dinov2Model.from_pretrained(
        str(unified.DINO_DIR), local_files_only=True, use_safetensors=True
    ).eval()

    mask = build_reliability_masks.load_mask(run, camera, unified.VARIANT.mask_variant)
    patch_valid = torch.tensor(mask["patch_valid"].reshape(-1).copy(), dtype=torch.bool)
    valid_count = int(patch_valid.sum())
    if valid_count == 0:
        raise ValueError(f"{run}/{camera}: reliability mask leaves no valid patch")

    total = unified.frame_count(run, camera)
    tokens = np.zeros((total, valid_count, model.config.hidden_size), dtype=TOKEN_DTYPE)

    batch_images: list[Image.Image] = []
    batch_ordinals: list[int] = []

    def flush() -> None:
        if not batch_images:
            return
        with torch.inference_mode():
            inputs = processor(
                images=batch_images, return_tensors="pt", do_resize=True,
                size={"height": unified.MODEL_SIZE, "width": unified.MODEL_SIZE}, do_center_crop=False,
            )
            hidden = model(**inputs).last_hidden_state[:, 1:, :]
            normalised = torch.nn.functional.normalize(hidden, dim=-1)[:, patch_valid, :]
        for index, ordinal in enumerate(batch_ordinals):
            tokens[ordinal] = normalised[index].numpy().astype(TOKEN_DTYPE)
        batch_images.clear()
        batch_ordinals.clear()

    for ordinal, image in frame_service.iter_frames(
        frame_service.source_path(run, camera), *frame_service.DESCRIPTOR_SIZE
    ):
        picture = Image.fromarray(image)
        if unified.VARIANT.greyscale:
            picture = picture.convert("L").convert("RGB")
        batch_images.append(picture)
        batch_ordinals.append(ordinal)
        if len(batch_images) == unified.DESCRIPTOR_BATCH:
            flush()
            if ordinal % 400 < unified.DESCRIPTOR_BATCH:
                print(f"   {run}/{camera} tokens: {ordinal + 1}/{total}", flush=True)
    flush()

    unified.atomic_write(target, lambda path: np.save(path, tokens))
    return np.load(target, mmap_mode="r")


def kmeans(samples: np.ndarray, k: int, iterations: int, seed: int) -> np.ndarray:
    """Plain Lloyd k-means on L2-normalised samples; deterministic given the seed."""

    rng = np.random.default_rng(seed)
    # k-means++ seeding: each new centre is drawn with probability proportional
    # to its squared distance from the nearest existing centre, so two seeds
    # cannot land in one cluster and leave another without a word.
    centres = np.empty((k, samples.shape[1]), dtype=np.float32)
    centres[0] = samples[rng.integers(samples.shape[0])]
    nearest = np.full(samples.shape[0], np.inf, dtype=np.float32)
    for index in range(1, k):
        distance = 2.0 - 2.0 * (samples @ centres[index - 1])   # squared distance of unit vectors
        nearest = np.minimum(nearest, distance)
        weights = np.maximum(nearest, 0.0)
        total = float(weights.sum())
        pick = rng.choice(samples.shape[0], p=weights / total) if total > 0 else rng.integers(samples.shape[0])
        centres[index] = samples[pick]
    for _ in range(iterations):
        # Cosine geometry: samples and centres are unit vectors, so the nearest
        # centre by dot product is the nearest by Euclidean distance.
        assignment = np.argmax(samples @ centres.T, axis=1)
        for index in range(k):
            members = samples[assignment == index]
            if len(members):
                centre = members.mean(axis=0)
                centres[index] = centre / max(np.linalg.norm(centre), 1e-8)
            else:
                centres[index] = samples[rng.integers(samples.shape[0])]
    return centres


def build_vocabulary(camera: str, force: bool = False) -> np.ndarray:
    """One vocabulary per camera from both runs, so the words are shared across the pair."""

    target = vocabulary_path(camera)
    if target.is_file() and not force:
        return np.load(target)
    pooled = []
    for run in ("runA", "runB"):
        tokens = extract_tokens(run, camera)
        sample = np.asarray(tokens[::VOCABULARY_FRAME_STRIDE], dtype=np.float32)
        pooled.append(sample.reshape(-1, sample.shape[-1]))
    samples = np.concatenate(pooled)
    print(f"   {camera}: k-means k={VOCABULARY_SIZE} on {samples.shape[0]} tokens", flush=True)
    centres = kmeans(samples, VOCABULARY_SIZE, KMEANS_ITERATIONS, KMEANS_SEED)
    import build_task2_unified as unified

    unified.atomic_write(target, lambda path: np.save(path, centres))
    return centres


def vlad(tokens: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Hard-assignment VLAD with intra-normalisation, then L2: [frames, k*dim]."""

    frames, _, dim = tokens.shape
    k = centres.shape[0]
    output = np.zeros((frames, k * dim), dtype=np.float32)
    for index in range(frames):
        frame_tokens = np.asarray(tokens[index], dtype=np.float32)
        assignment = np.argmax(frame_tokens @ centres.T, axis=1)
        residual = frame_tokens - centres[assignment]
        blocks = np.zeros((k, dim), dtype=np.float32)
        np.add.at(blocks, assignment, residual)
        norms = np.linalg.norm(blocks, axis=1, keepdims=True)
        blocks = blocks / np.maximum(norms, 1e-8)          # intra-normalisation
        vector = blocks.reshape(-1)
        output[index] = vector / max(np.linalg.norm(vector), 1e-8)
    return output


def descriptors(run: str, camera: str, target: Path, force: bool = False) -> np.ndarray:
    """VLAD descriptors for every frame, cached at `target` (the unified cache path)."""

    if target.is_file() and not force:
        return np.load(target)
    import build_task2_unified as unified

    tokens = extract_tokens(run, camera, force=False)
    centres = build_vocabulary(camera)
    print(f"   {run}/{camera}: VLAD over {tokens.shape[0]} frames", flush=True)
    output = vlad(tokens, centres)
    unified.atomic_write(target, lambda path: np.save(path, output))
    return output
