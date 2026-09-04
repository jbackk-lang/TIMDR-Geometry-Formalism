"""
tests/test_chronocongruence.py

Testy timdr_geometry.chronocongruence -- kongruencja Gamma(t,s) dla
chronoprocesu (galaz G). Dwie kategorie testow:

  1. Rownowaznosc: make_congruence_mesh z odpowiednim gamma i tymi
     samymi wartosciami t/s co make_plane_mesh/make_cylinder_mesh/
     make_sphere_mesh MUSI dac dokladnie te same wierzcholki i trojkaty
     co dedykowane generatory w weingarten.py -- to jest test
     DETERMINISTYCZNY (bez tolerancji na wierzcholkach/trojkatach poza
     zaokragleniem float), niezalezny od tego, czy krzywizna wychodzi
     poprawnie (to sprawdza test_geometry_weingarten.py).
  2. Bezposrednia krzywizna na kongruencji walcowej, zbudowanej WPROST
     przez make_congruence_mesh (nie przez make_cylinder_mesh) -- ten
     sam fakt analityczny (jedna krzywizna ~0, druga ~1/r), ale
     niezaleznie od tego, czy make_cylinder_mesh jest poprawne.

UWAGA: ten plik NIE zostal uruchomiony w sesji, w ktorej powstal
(sandbox bash niedostepny). Tolerancje sa swiadomie te same, co juz
ustalone w tests/test_geometry_weingarten.py dla identycznych
przypadkow promien/rozdzielczosc. Uruchom `pytest tests/ -v`.
"""
import numpy as np
import pytest

from timdr_geometry import (
    vertex_normals,
    one_ring,
    discrete_shape_operator,
    make_plane_mesh,
    make_sphere_mesh,
    make_cylinder_mesh,
)
from timdr_geometry.chronocongruence import (
    make_congruence_mesh,
    flat_parallel_congruence,
    cylindrical_congruence,
    spherical_congruence,
)


def _interior_rows(n_rows: int, n_cols: int, margin: int) -> list:
    """Ta sama definicja co w tests/test_geometry_weingarten.py --
    dziala identycznie tutaj, bo make_congruence_mesh uzywa tego samego
    row-major indeksowania (t=wiersz, s=kolumna)."""
    return [j * n_cols + i for j in range(margin, n_rows - margin) for i in range(n_cols)]


# ---------------------------------------------------------------------
# Rownowaznosc z dedykowanymi generatorami siatek
# ---------------------------------------------------------------------

class TestEquivalenceToExistingGenerators:
    def test_flat_congruence_matches_make_plane_mesh(self):
        pts = np.linspace(0.0, 1.0, 8)
        via_congruence = make_congruence_mesh(
            flat_parallel_congruence, t_values=pts, s_values=pts,
            t_periodic=False, s_periodic=False,
        )
        via_dedicated = make_plane_mesh(n=8, size=1.0)
        np.testing.assert_allclose(via_congruence.vertices, via_dedicated.vertices, atol=1e-12)
        np.testing.assert_array_equal(via_congruence.faces, via_dedicated.faces)

    def test_cylindrical_congruence_matches_make_cylinder_mesh(self):
        radius, n_theta, n_z, height = 1.5, 24, 12, 3.0
        zs = np.linspace(0.0, height, n_z)
        thetas = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)

        via_congruence = make_congruence_mesh(
            lambda t, s: cylindrical_congruence(t, s, radius=radius),
            t_values=zs, s_values=thetas,
            t_periodic=False, s_periodic=True,
        )
        via_dedicated = make_cylinder_mesh(n_theta, n_z, radius=radius, height=height)

        np.testing.assert_allclose(via_congruence.vertices, via_dedicated.vertices, atol=1e-10)
        np.testing.assert_array_equal(via_congruence.faces, via_dedicated.faces)

    def test_spherical_congruence_matches_make_sphere_mesh(self):
        radius, n_lat, n_lon = 2.0, 24, 24
        lats = np.linspace(np.pi / n_lat, np.pi - np.pi / n_lat, n_lat)
        lons = np.linspace(0.0, 2 * np.pi, n_lon, endpoint=False)

        via_congruence = make_congruence_mesh(
            lambda t, s: spherical_congruence(t, s, radius=radius),
            t_values=lats, s_values=lons,
            t_periodic=False, s_periodic=True,
        )
        via_dedicated = make_sphere_mesh(n_lat, n_lon, radius=radius)

        np.testing.assert_allclose(via_congruence.vertices, via_dedicated.vertices, atol=1e-10)
        np.testing.assert_array_equal(via_congruence.faces, via_dedicated.faces)


# ---------------------------------------------------------------------
# Bezposrednia krzywizna na kongruencji zbudowanej WPROST przez
# make_congruence_mesh (niezaleznie od make_cylinder_mesh)
# ---------------------------------------------------------------------

class TestCylindricalCongruenceCurvatureDirect:
    RADIUS = 1.5
    N_THETA = 24
    N_Z = 12
    HEIGHT = 3.0

    def test_axial_curvature_zero_circumferential_matches_1_over_r(self):
        zs = np.linspace(0.0, self.HEIGHT, self.N_Z)
        thetas = np.linspace(0.0, 2 * np.pi, self.N_THETA, endpoint=False)
        mesh = make_congruence_mesh(
            lambda t, s: cylindrical_congruence(t, s, radius=self.RADIUS),
            t_values=zs, s_values=thetas,
            t_periodic=False, s_periodic=True,
        )
        normals = vertex_normals(mesh)
        rings = one_ring(mesh)
        expected = 1.0 / self.RADIUS
        interior = _interior_rows(self.N_Z, self.N_THETA, margin=2)

        for idx in interior:
            op = discrete_shape_operator(mesh, normals, idx, rings)
            kappas = np.sort(np.abs(op.principal_curvatures))
            assert kappas[0] < 0.1, f"oczekiwano ~0 wzdluz t (osiowo), dostano {kappas[0]}"
            assert abs(kappas[1] - expected) < 0.25 * expected, (
                f"oczekiwano ~{expected} wzdluz s (obwodowo), dostano {kappas[1]}"
            )


class TestFlatCongruenceCurvatureDirect:
    N = 8

    def test_shape_operator_is_zero(self):
        pts = np.linspace(0.0, 1.0, self.N)
        mesh = make_congruence_mesh(
            flat_parallel_congruence, t_values=pts, s_values=pts,
            t_periodic=False, s_periodic=False,
        )
        normals = vertex_normals(mesh)
        rings = one_ring(mesh)
        interior = _interior_rows(self.N, self.N, margin=1)
        for idx in interior:
            op = discrete_shape_operator(mesh, normals, idx, rings)
            assert np.max(np.abs(op.principal_curvatures)) < 1e-8


# ---------------------------------------------------------------------
# Przypadki brzegowe make_congruence_mesh
# ---------------------------------------------------------------------

class TestEdgeCases:
    def test_rejects_too_few_t_values(self):
        with pytest.raises(ValueError):
            make_congruence_mesh(flat_parallel_congruence, t_values=[0.0], s_values=[0.0, 1.0])

    def test_rejects_too_few_s_values(self):
        with pytest.raises(ValueError):
            make_congruence_mesh(flat_parallel_congruence, t_values=[0.0, 1.0], s_values=[0.0])

    def test_rejects_gamma_with_wrong_output_shape(self):
        bad_gamma = lambda t, s: np.array([t, s])  # tylko 2D, nie 3D
        with pytest.raises(ValueError):
            make_congruence_mesh(bad_gamma, t_values=[0.0, 1.0], s_values=[0.0, 1.0])
