#!/usr/bin/env python3
"""Tests for the unsupervised VLAD aggregation variant."""

from __future__ import annotations

import unittest

import numpy as np

import build_task2_unified as unified
import vlad_aggregation as va


def unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True)


class KMeansTests(unittest.TestCase):
    def test_recovers_well_separated_clusters(self):
        rng = np.random.default_rng(0)
        centres = unit(rng.normal(size=(4, 16)))
        samples = unit(np.concatenate([c + 0.05 * rng.normal(size=(200, 16)) for c in centres]))
        found = va.kmeans(samples, 4, iterations=20, seed=1)
        similarity = found @ centres.T
        self.assertTrue(np.all(similarity.max(axis=1) > 0.98))
        self.assertEqual(len(set(similarity.argmax(axis=1))), 4)

    def test_deterministic_given_seed(self):
        rng = np.random.default_rng(3)
        samples = unit(rng.normal(size=(300, 8)))
        a = va.kmeans(samples, 5, 10, seed=7)
        b = va.kmeans(samples, 5, 10, seed=7)
        np.testing.assert_array_equal(a, b)


class VladTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.centres = unit(rng.normal(size=(6, 12)))
        self.rng = rng

    def test_output_is_unit_length_and_right_shape(self):
        tokens = unit(self.rng.normal(size=(3, 40, 12))).astype(np.float16)
        out = va.vlad(tokens, self.centres)
        self.assertEqual(out.shape, (3, 6 * 12))
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_small_distinct_structure_survives_where_a_mean_loses_it(self):
        # 95% of tokens are "road" near word 0; 5% are a distinctive structure
        # near word 3. Two frames differ only in whether the structure is
        # present. VLAD must separate them more than the mean does.
        road = self.centres[0]
        sign = self.centres[3]
        n = 200
        base = unit(road + 0.02 * self.rng.normal(size=(n, 12)))
        with_sign = base.copy()
        with_sign[:10] = unit(sign + 0.02 * self.rng.normal(size=(10, 12)))
        without = base.copy()
        without[:10] = unit(road + 0.02 * self.rng.normal(size=(10, 12)))
        tokens = np.stack([with_sign, without]).astype(np.float16)
        v = va.vlad(tokens, self.centres)
        mean = unit(tokens.astype(np.float32).mean(axis=1))
        vlad_similarity = float(v[0] @ v[1])
        mean_similarity = float(mean[0] @ mean[1])
        self.assertLess(vlad_similarity, mean_similarity)
        self.assertGreater(mean_similarity, 0.95)

    def test_identical_frames_have_similarity_one(self):
        tokens = unit(self.rng.normal(size=(1, 30, 12))).astype(np.float16)
        v = va.vlad(np.concatenate([tokens, tokens]), self.centres)
        self.assertAlmostEqual(float(v[0] @ v[1]), 1.0, places=5)


class PlumbingTests(unittest.TestCase):
    def test_default_aggregation_is_the_shipped_mean(self):
        self.assertEqual(unified.VARIANT.aggregation, "mean")
        self.assertEqual(unified.variant_name(), "")
        self.assertEqual(unified.descriptor_version(), unified.DESCRIPTOR_VERSION)

    def test_vlad_routes_to_its_own_directory_and_cache_name(self):
        saved = unified.configure_variant(aggregation="vlad")
        try:
            self.assertEqual(unified.variant_name(), "vlad")
            self.assertTrue(unified.descriptor_version().endswith(f"-vlad{va.VOCABULARY_SIZE}"))
            self.assertEqual(unified.output_dir("cam5").parent.parent, unified.VARIANT_OUTPUT_DIR)
        finally:
            unified.set_variant(saved)


if __name__ == "__main__":
    unittest.main()
