"""
tests/test_geometry_weingarten.py

Testy stabilności dyskretnego operatora Weingartena (Aksjomaty G8-G9,
`GIA-TIMDR/docs/theory/Axioms_G_TIMDR_Geometry.md`) na trzech
przypadkach znanych analitycznie (płaszczyzna, sfera, walec) plus test
zbieżności przy zagęszczaniu siatki. Dokładnie te cztery testy, które
Aksjomat G7c wymienia jako brakujące (implementacja numeryczna + jej
weryfikacja) — z zastrzeżeniem, że to weryfikacja NA PRZYPADKACH
SYNTETYCZNYCH o znanej odpowiedzi, nie na rzeczywistych danych
geometrycznych (G7c(3) pozostaje otwarte).

UWAGA UCZCIWOŚCIOWA: ten plik NIE był uruchomiony w tej sesji — sandbox
bash był niedostępny (RPC pipe closed) przez cały czas pisania tego
modułu. Matematyka dopasowania (rzut styczny, MNK, symetryzacja,
eigendekompozycja) została prześledzona ręcznie krok po kroku i wygląda
poprawnie, a tolerancje liczbowe poniżej są celowo szerokie, żeby nie
polegać na precyzyjnym zgadywaniu błędu dyskretyzacji bez wykonania
kodu — ale to NIE zastępuje faktycznego uruchomienia `pytest tests/ -v`.
Zrób to przed zaufaniem tym liczbom, dokładnie jak
`examples/real_weather_resonance_validation.py` w TIMDR-Math-Formalism
było oznaczone tym samym zastrzeżeniem z tego samego powodu.
"""
import numpy as np
import pytest

from timdr_geometry import (
    Mesh,
    vertex_normals,
    one_ring,
    discrete_shape_operator,
    T_S_empirical,
    T_S_predicted,
    make_plane_mesh,
    make_sphere_mesh,
    make_cylinder_mesh,
)


def _interior_rows(n_rows: int, n_cols: int, margin: int) -> list:
    """Indeksy wierzchołków w wierszach [margin, n_rows-margin) siatki
    n_rows x n_cols (row-major) — pomija wiersze blisko brzegu/biegunów,
    gdzie 1-ring jest niepełny lub siatka lat-long jest zdegenerowana."""
    return [j * n_cols + i for j in range(margin, n_rows - margin) for i in range(n_cols)]


# ---------------------------------------------------------------------
# Test 2 (kolejność z zadania) — płaszczyzna: S_p≈0, T_S≈0
# ---------------------------------------------------------------------

class TestPlane:
    N = 8

    def test_normals_are_constant(self):
        mesh = make_plane_mesh(n=self.N, size=1.0)
        normals = vertex_normals(mesh)
        assert np.allclose(normals, normals[0], atol=1e-10)
        assert np.allclose(np.abs(normals[0]), [0.0, 0.0, 1.0], atol=1e-10)

    def test_shape_operator_is_zero(self):
        mesh = make_plane_mesh(n=self.N, size=1.0)
        normals = vertex_normals(mesh)
        rings = one_ring(mesh)
        interior = _interior_rows(self.N, self.N, margin=1)
        for idx in interior:
            op = discrete_shape_operator(mesh, normals, idx, rings)
            assert np.max(np.abs(op.principal_curvatures)) < 1e-8

    def test_T_S_is_zero_for_every_edge(self):
        mesh = make_plane_mesh(n=self.N, size=1.0)
        normals = vertex_normals(mesh)
        rings = one_ring(mesh)
        interior = _interior_rows(self.N, self.N, margin=1)
        for idx in interior:
            for q_idx in rings[idx]:
                assert T_S_empirical(mesh, normals, idx, q_idx) < 1e-10


# ---------------------------------------------------------------------
# Test 1 — sfera: obie krzywizny główne ≈ 1/R (magnituda, patrz uwaga
# o konwencji znaku w weingarten.py)
# ---------------------------------------------------------------------

class TestSphere:
    RADIUS = 2.0
    N_LAT = 24
    N_LON = 24
    MARGIN = 3

    def test_principal_curvatures_match_1_over_R(self):
        mesh = make_sphere_mesh(self.N_LAT, self.N_LON, radius=self.RADIUS)
        normals = vertex_normals(mesh)
        rings = one_ring(mesh)
        expected = 1.0 / self.RADIUS
        interior = _interior_rows(self.N_LAT, self.N_LON, margin=self.MARGIN)

        errors = []
        for idx in interior:
            op = discrete_shape_operator(mesh, normals, idx, rings)
            kappas = np.abs(op.principal_curvatures)
            errors.append(np.max(np.abs(kappas - expected)))
        errors = np.array(errors)

        # Tolerancja szeroka celowo (kod nie był uruchomiony, patrz
        # zastrzeżenie na górze pliku) — sprawdza rząd wielkości błędu
        # dyskretyzacyjnego dla siatki 24x24 na promieniu 2, nie
        # precyzyjną wartość.
        assert np.median(errors) < 0.15
        assert np.max(errors) < 0.4

    def test_gaussian_curvature_positive_and_matches_1_over_R2(self):
        mesh = make_sphere_mesh(self.N_LAT, self.N_LON, radius=self.RADIUS)
        normals = vertex_normals(mesh)
        rings = one_ring(mesh)
        expected_K = 1.0 / (self.RADIUS ** 2)
        interior = _interior_rows(self.N_LAT, self.N_LON, margin=self.MARGIN)

        Ks = []
        for idx in interior:
            op = discrete_shape_operator(mesh, normals, idx, rings)
            # K = kappa1*kappa2; ze wzgledu na konwencje znaku obie
            # kappy maja ten sam znak dla wypuklej powierzchni, wiec K>0
            Ks.append(float(np.prod(op.principal_curvatures)))
        Ks = np.array(Ks)

        assert np.all(Ks > 0.0)
        assert abs(np.median(Ks) - expected_K) < 0.5 * expected_K


# ---------------------------------------------------------------------
# Test 3 — walec: jedna krzywizna ≈0 (oś), druga ≈1/r (obwód)
# ---------------------------------------------------------------------

class TestCylinder:
    RADIUS = 1.5
    N_THETA = 24
    N_Z = 12

    def test_one_curvature_zero_other_matches_1_over_r(self):
        mesh = make_cylinder_mesh(
            self.N_THETA, self.N_Z, radius=self.RADIUS, height=3.0
        )
        normals = vertex_normals(mesh)
        rings = one_ring(mesh)
        expected = 1.0 / self.RADIUS
        interior = _interior_rows(self.N_Z, self.N_THETA, margin=2)

        for idx in interior:
            op = discrete_shape_operator(mesh, normals, idx, rings)
            kappas = np.sort(np.abs(op.principal_curvatures))
            assert kappas[0] < 0.1, f"oczekiwano ~0 wzdłuż osi, dostano {kappas[0]}"
            assert abs(kappas[1] - expected) < 0.25 * expected, (
                f"oczekiwano ~{expected} obwodowo, dostano {kappas[1]}"
            )


# ---------------------------------------------------------------------
# Test 4 — zbieżność przy zagęszczaniu siatki
# ---------------------------------------------------------------------

class TestMeshRefinement:
    def test_sphere_curvature_error_decreases_with_resolution(self):
        radius = 1.0
        expected = 1.0 / radius
        resolutions = [(10, 10), (20, 20), (36, 36)]
        mean_errors = []

        for n_lat, n_lon in resolutions:
            mesh = make_sphere_mesh(n_lat, n_lon, radius=radius)
            normals = vertex_normals(mesh)
            rings = one_ring(mesh)
            margin = max(2, n_lat // 6)
            interior = _interior_rows(n_lat, n_lon, margin=margin)

            errs = []
            for idx in interior:
                op = discrete_shape_operator(mesh, normals, idx, rings)
                kappas = np.abs(op.principal_curvatures)
                errs.append(np.max(np.abs(kappas - expected)))
            mean_errors.append(float(np.mean(errs)))

        # Zbieżność: błąd przy najgęstszej siatce wyraźnie mniejszy niż
        # przy najrzadszej. Nie wymagamy monotoniczności na każdym kroku
        # (dyskretyzacja bywa nie-monotoniczna lokalnie), tylko wyraźnej
        # poprawy między skrajami.
        assert mean_errors[-1] < mean_errors[0]
        assert mean_errors[-1] < 0.5 * mean_errors[0]

    def test_T_S_predicted_matches_empirical_better_at_higher_resolution(self):
        radius = 2.0
        resolutions = [(12, 12), (30, 30)]
        median_rel_errors = []

        for n_lat, n_lon in resolutions:
            mesh = make_sphere_mesh(n_lat, n_lon, radius=radius)
            normals = vertex_normals(mesh)
            rings = one_ring(mesh)
            margin = max(2, n_lat // 6)
            interior = _interior_rows(n_lat, n_lon, margin=margin)

            rel_errors = []
            for idx in interior:
                op = discrete_shape_operator(mesh, normals, idx, rings)
                for q_idx in sorted(rings[idx]):
                    delta_p = mesh.vertices[q_idx] - mesh.vertices[idx]
                    predicted = T_S_predicted(op, delta_p)
                    actual = T_S_empirical(mesh, normals, idx, q_idx)
                    if actual > 1e-9:
                        rel_errors.append(abs(predicted - actual) / actual)
            median_rel_errors.append(float(np.median(rel_errors)))

        # Aksjomat G9c: T_S = F(W_S) + O(||Δp||^2), więc błąd względny
        # (rzędu O(||Δp||)) powinien maleć przy gęstszej siatce.
        assert median_rel_errors[-1] < median_rel_errors[0]


# ---------------------------------------------------------------------
# Przypadki brzegowe
# ---------------------------------------------------------------------

class TestEdgeCases:
    def test_isolated_vertex_raises(self):
        verts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 5.0, 5.0]]
        )
        faces = np.array([[0, 1, 2]])
        mesh = Mesh(verts, faces)
        normals = vertex_normals(mesh)
        rings = one_ring(mesh)
        with pytest.raises(ValueError):
            discrete_shape_operator(mesh, normals, 3, rings)

    def test_minimum_two_neighbors_does_not_raise(self):
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        mesh = Mesh(verts, faces)
        normals = vertex_normals(mesh)
        rings = one_ring(mesh)
        op = discrete_shape_operator(mesh, normals, 0, rings)
        assert op.n_neighbors == 2
        # jeden trójkąt -> normalne stałe wszędzie -> S_p dokładnie 0
        assert np.allclose(op.matrix_2d, 0.0, atol=1e-10)
