#!/usr/bin/env python3
"""Tests for the tolerance-curve reading of the sealed blind sets."""

from __future__ import annotations

import unittest

import evaluate_tolerance_curve as tol


class CurveTests(unittest.TestCase):
    def setUp(self):
        self.labels = [
            {"runA_frame": "1", "runB_min": "10", "runB_max": "10"},
            {"runA_frame": "2", "runB_min": "20", "runB_max": "22"},
            {"runA_frame": "3", "runB_min": "30", "runB_max": "30"},
            {"runA_frame": "4", "runB_min": "40", "runB_max": "40"},
        ]

    def test_fractions_are_monotone_in_tolerance(self):
        mapping = {1: "10", 2: "24", 3: "36", 4: ""}      # distances 0, 2, 6; one abstention
        result = tol.curve(self.labels, mapping)
        self.assertEqual(result["accepted"], 3)
        self.assertEqual(result["coverage"], 0.75)
        fractions = [result["within"][str(t)]["fraction"] for t in tol.TOLERANCES]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(result["within"]["0"]["hits"], 1)
        self.assertEqual(result["within"]["2"]["hits"], 2)
        self.assertEqual(result["within"]["5"]["hits"], 2)
        self.assertEqual(result["within"]["10"]["hits"], 3)

    def test_interval_membership_is_a_strict_hit(self):
        mapping = {1: "10", 2: "21", 3: "30", 4: "40"}
        result = tol.curve(self.labels, mapping)
        self.assertEqual(result["within"]["0"]["fraction"], 1.0)

    def test_no_acceptance_gives_no_fraction(self):
        result = tol.curve(self.labels, {1: "", 2: "", 3: "", 4: ""})
        self.assertEqual(result["accepted"], 0)
        self.assertIsNone(result["within"]["2"]["fraction"])

    def test_headline_tolerance_is_declared_and_in_the_grid(self):
        self.assertIn(tol.HEADLINE_TOLERANCE, tol.TOLERANCES)
        self.assertEqual(tol.HEADLINE_TOLERANCE, 2)


if __name__ == "__main__":
    unittest.main()
