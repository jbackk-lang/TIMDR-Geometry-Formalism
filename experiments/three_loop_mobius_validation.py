#!/usr/bin/env python3
"""Reproducible checks for the proposed three-loop model (not experimental validation).

Run: python experiments/three_loop_mobius_validation.py
Uses only Python standard library.
"""
import cmath
import math
import unittest

def matvec(a, x):
    return tuple(sum(a[i][j] * x[j] for j in range(3)) for i in range(3))

def norm(x):
    return math.sqrt(sum(abs(z) ** 2 for z in x))

def j_matrix(lam):
    return ((0., 1., -lam), (-lam, 0., 1.), (1., -lam, 0.))

def spectral_radius(lam):
    return max(abs(1-lam), math.sqrt(1+lam+lam*lam))

def stable_matrix(lam, gamma):
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    scale = math.exp(-gamma)/spectral_radius(lam)
    return tuple(tuple(scale * x for x in row) for row in j_matrix(lam))

def torus(theta, phi, R=2., r=.5):
    return ((R+r*math.cos(theta))*math.cos(phi),
            (R+r*math.cos(theta))*math.sin(phi), r*math.sin(theta))

def original_embedding(theta, phi, R=2., r=.5):
    return ((R+r*math.cos(1.5*theta))*math.cos(phi),
            (R+r*math.cos(1.5*theta))*math.sin(phi),
            r*math.sin(1.5*theta))

def mobius(u, v, R=2.):
    return ((R+v*math.cos(u/2))*math.cos(u),
            (R+v*math.cos(u/2))*math.sin(u),
            v*math.sin(u/2))

def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

class ThreeLoopTests(unittest.TestCase):
    def test_eigenvalues(self):
        for lam in (-2., -0.5, 0., .2, 1., 2.):
            omega=cmath.exp(2j*math.pi/3)
            eig=[omega**k-lam*omega**(2*k) for k in range(3)]
            self.assertAlmostEqual(max(map(abs,eig)), spectral_radius(lam), places=12)

    def test_original_positive_lambda_is_not_damping(self):
        for lam in (.01, .2, 1., 3.):
            self.assertGreater(spectral_radius(lam), 1.)

    def test_stabilized_mode_bounds(self):
        omega=cmath.exp(2j*math.pi/3)
        for lam in (-1., 0., .3, 2.):
            for gamma in (0., .2, 1.):
                a=stable_matrix(lam,gamma)
                for k in range(3):
                    x=tuple(omega**(k*i) for i in range(3))
                    self.assertLessEqual(norm(matvec(a,x))/norm(x),math.exp(-gamma)+1e-12)

    def test_mobius_edge_identification(self):
        for v in (-.4, -.1, 0., .2, .4):
            self.assertLess(distance(mobius(0.,v),mobius(2*math.pi,-v)),1e-12)

    def test_original_embedding_is_torus_reparametrization(self):
        for theta in (0., .3, 1., 2.5, 4.):
            for phi in (.1, 1., 3.):
                self.assertLess(distance(original_embedding(theta,phi),torus(1.5*theta,phi)),1e-12)

if __name__=="__main__":
    unittest.main(verbosity=2)
