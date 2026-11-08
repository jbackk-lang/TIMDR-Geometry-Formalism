#!/usr/bin/env python3
"""Three-channel transport under four proposed gluing models.

Mathematical toy simulator, not empirical validation.
Run: python experiments/holonomy_simulator.py
Tests: python -m unittest discover -s experiments -p holonomy_simulator.py -v
"""
import math
import unittest


def transport(state, model, circuits):
    """Return (transverse_orientation_sign, channel_state) after circuits."""
    if model not in ("M0", "M1", "M2", "M3"):
        raise ValueError("model must be M0, M1, M2 or M3")
    if not isinstance(circuits, int) or circuits < 0:
        raise ValueError("circuits must be a nonnegative integer")
    if len(state) != 3:
        raise ValueError("three channels required")
    z = tuple(complex(x) for x in state)
    orientation = 1
    for _ in range(circuits):
        if model != "M0":
            orientation *= -1
        if model == "M2":
            z = (z[1], z[2], z[0])
        elif model == "M3":
            z = tuple(-x for x in z)
    return orientation, z


def j_action(state, lam):
    a, b, c = state
    return (b - lam * c, c - lam * a, a - lam * b)


def stabilized_action(state, lam, gamma):
    if not math.isfinite(lam) or not math.isfinite(gamma) or gamma < 0:
        raise ValueError("lam finite; gamma finite and >= 0")
    radius = max(abs(1 - lam), math.sqrt(1 + lam + lam * lam))
    return tuple(math.exp(-gamma) * x / radius for x in j_action(state, lam))


def squared_norm(state):
    return sum(abs(x) ** 2 for x in state)


class Tests(unittest.TestCase):
    def setUp(self):
        self.z = (1 + 0j, 2 + 1j, -1 + 2j)

    def test_orientability(self):
        self.assertEqual(transport(self.z, "M0", 1)[0], 1)
        for m in ("M1", "M2", "M3"):
            self.assertEqual(transport(self.z, m, 1)[0], -1)
            self.assertEqual(transport(self.z, m, 2)[0], 1)

    def test_channel_periods(self):
        self.assertEqual(transport(self.z, "M2", 3)[1], self.z)
        self.assertNotEqual(transport(self.z, "M2", 1)[1], self.z)
        self.assertEqual(transport(self.z, "M3", 2)[1], self.z)
        self.assertEqual(transport(self.z, "M1", 1)[1], self.z)
        self.assertEqual(transport(self.z, "M2", 6), (1, self.z))

    def test_commutation(self):
        for lam in (-1., 0., .4, 2.):
            for m in ("M0", "M1", "M2", "M3"):
                lhs = transport(j_action(self.z, lam), m, 1)[1]
                rhs = j_action(transport(self.z, m, 1)[1], lam)
                self.assertTrue(all(abs(a - b) < 1e-12 for a, b in zip(lhs, rhs)))

    def test_contraction(self):
        for lam in (-2., -.2, 0., .2, 3.):
            for gamma in (.1, .5, 1.):
                out = stabilized_action(self.z, lam, gamma)
                self.assertLessEqual(squared_norm(out), math.exp(-2 * gamma) * squared_norm(self.z) + 1e-12)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            transport(self.z, "M4", 1)
        with self.assertRaises(ValueError):
            transport(self.z, "M1", -1)


def demo():
    initial = (1 + 0j, 2 + 1j, -1 + 2j)
    print("Model | circuits | orientation | channel state")
    for m in ("M0", "M1", "M2", "M3"):
        for n in (0, 1, 2, 3, 6):
            sign, state = transport(initial, m, n)
            print(f"{m:5} | {n:8} | {sign:+11} | {state}")


if __name__ == "__main__":
    demo()
