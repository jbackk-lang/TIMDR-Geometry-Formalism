"""
tests/test_envelope.py

Testy Aksjomatu G10 (obwiednia zaokrąglona ∂_R(Δ), parametr (P,Q)) —
`GIA-TIMDR/docs/theory/Axioms_G_TIMDR_Geometry.md`. Skupione na
NUMERYCZNEJ weryfikacji tożsamości, które G10c-e wyprowadzają ręcznie —
w tym w szczególności poprawionej wersji G10e (patrz moduł
`timdr_geometry/envelope.py`, akapit "WAŻNE — poprawka błędu..."):
`R_max(Δ)=r_in(Δ)` DOKŁADNIE dla każdego trójkąta, nie tylko
równobocznego. TestRMaxEqualsRIn poniżej weryfikuje to na trójkącie
ostrym, prostokątnym (3-4-5, dokładne wartości z ręcznego rachunku) i
prawie zdegenerowanym — dwiema niezależnymi ścieżkami (suma kotangensów
połówek kątów vs wzór Herona), żeby test nie był kołowy.

UWAGA UCZCIWOŚCIOWA: ten plik został odtąd faktycznie uruchomiony przez
użytkownika (`pytest tests/test_envelope.py -v`) i ZWERYFIKOWANY —
**65/65 testów przeszło**, w tym wszystkie warianty na 5 trójkątach
testowych (poprawka G10e, monotoniczność, odwracalność, niezależna
weryfikacja konstrukcji geometrycznej).
"""
import numpy as np
import pytest

from timdr_geometry import (
    TriangleGeometry,
    L0_of_R,
    Lk_of_R,
    L_of_R,
    P_of_R,
    Q_of_R,
    R_of_P,
    rounded_triangle_boundary,
    boundary_length_numeric,
    verify_envelope_length,
)


# Kilka trójkątów testowych: (nazwa, boki a,b,c)
TRIANGLES = {
    "3-4-5 (prostokątny, jawnie asymetryczny)": (3.0, 4.0, 5.0),
    "równoboczny (a=2)": (2.0, 2.0, 2.0),
    "4-5-6 (ostry, mocno skalenowy)": (4.0, 5.0, 6.0),
    "rozwarty 2-2-3.9": (2.0, 2.0, 3.9),
    "prawie zdegenerowany 1-1-1.99": (1.0, 1.0, 1.99),
}


# ---------------------------------------------------------------------
# TriangleGeometry — podstawowe wielkości
# ---------------------------------------------------------------------

class TestTriangleGeometry345:
    """Wartości z ręcznego rachunku (ta sama sesja, użyte do znalezienia
    błędu w G10e): boki 3-4-5, P0=12, c(Δ)=6.0, s=6, area=6, r_in=1.0."""

    def setup_method(self):
        self.tri = TriangleGeometry.from_sides(3.0, 4.0, 5.0)

    def test_perimeter_and_semiperimeter(self):
        assert self.tri.P0 == pytest.approx(12.0)
        assert self.tri.s == pytest.approx(6.0)

    def test_area_heron(self):
        assert self.tri.area == pytest.approx(6.0)

    def test_r_in(self):
        assert self.tri.r_in == pytest.approx(1.0)

    def test_c_sum(self):
        # cot(A/2)+cot(B/2)+cot(C/2) = 3.0+2.0+1.0 = 6.0 (rachunek ręczny)
        assert self.tri.c_sum == pytest.approx(6.0, abs=1e-9)

    def test_from_points_matches_from_sides(self):
        tri2 = TriangleGeometry.from_points([0.0, 0.0], [3.0, 0.0], [0.0, 4.0])
        # bok naprzeciw (0,0)=A to |BC|=5, naprzeciw (3,0)=B to |CA|=4,
        # naprzeciw (0,4)=C to |AB|=3 -> sides=(5,4,3), permutacja tego
        # samego trójkąta 3-4-5, więc te same niezmienniki skalarne.
        assert tri2.P0 == pytest.approx(self.tri.P0)
        assert tri2.area == pytest.approx(self.tri.area)
        assert tri2.r_in == pytest.approx(self.tri.r_in)
        assert tri2.c_sum == pytest.approx(self.tri.c_sum)


class TestTriangleGeometryValidation:
    def test_rejects_non_positive_side(self):
        with pytest.raises(ValueError):
            TriangleGeometry.from_sides(0.0, 1.0, 1.0)

    def test_rejects_triangle_inequality_violation(self):
        with pytest.raises(ValueError):
            TriangleGeometry.from_sides(1.0, 1.0, 10.0)


# ---------------------------------------------------------------------
# Poprawka G10e: R_max(Δ) = r_in(Δ) DOKŁADNIE dla KAŻDEGO trójkąta —
# dwie niezależne ścieżki (cotangensy vs Heron), żeby nie było kołowe.
# ---------------------------------------------------------------------

class TestRMaxEqualsRIn:
    @pytest.mark.parametrize("name,sides", list(TRIANGLES.items()))
    def test_R_max_equals_r_in(self, name, sides):
        tri = TriangleGeometry.from_sides(*sides)
        assert tri.R_max() == pytest.approx(tri.r_in, rel=1e-9), (
            f"{name}: R_max={tri.R_max()} powinno równać się r_in={tri.r_in} "
            "dokładnie (poprawiona wersja G10e) — jeśli to pada, poprawka "
            "opisana w Axioms_G_TIMDR_Geometry.md jest błędna albo kod jej "
            "nie odzwierciedla poprawnie"
        )

    def test_345_exact_value(self):
        tri = TriangleGeometry.from_sides(3.0, 4.0, 5.0)
        assert tri.R_max() == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("name,sides", list(TRIANGLES.items()))
    def test_c_sum_equals_s_over_r_in(self, name, sides):
        """Druga tożsamość z tej samej poprawki: c(Δ)=s/r_in dokładnie."""
        tri = TriangleGeometry.from_sides(*sides)
        assert tri.c_sum == pytest.approx(tri.s / tri.r_in, rel=1e-9), name

    @pytest.mark.parametrize("name,sides", list(TRIANGLES.items()))
    def test_L0_at_R_max_is_zero_for_every_triangle(self, name, sides):
        """Poprawka G10e: obwiednia degeneruje się DOKŁADNIE do okręgu
        wpisanego przy R=R_max dla KAŻDEGO trójkąta, nie tylko
        równobocznego."""
        tri = TriangleGeometry.from_sides(*sides)
        r_max = tri.R_max()
        assert L0_of_R(tri, r_max) == pytest.approx(0.0, abs=1e-7), name

    @pytest.mark.parametrize("name,sides", list(TRIANGLES.items()))
    def test_Q_reaches_one_at_R_max_for_every_triangle(self, name, sides):
        tri = TriangleGeometry.from_sides(*sides)
        r_max = tri.R_max()
        assert Q_of_R(tri, r_max) == pytest.approx(1.0, abs=1e-7), name
        assert P_of_R(tri, r_max) == pytest.approx(0.0, abs=1e-7), name


class TestJensenInequality:
    """G10d: c(Δ) >= 3*sqrt(3), równość TYLKO dla równobocznego —
    baza monotoniczności L(R)."""

    def test_equilateral_achieves_equality(self):
        tri = TriangleGeometry.from_sides(2.0, 2.0, 2.0)
        assert tri.c_sum == pytest.approx(3.0 * np.sqrt(3.0), rel=1e-9)

    @pytest.mark.parametrize(
        "name,sides", [(n, s) for n, s in TRIANGLES.items() if n != "równoboczny (a=2)"]
    )
    def test_non_equilateral_strictly_exceeds_bound(self, name, sides):
        tri = TriangleGeometry.from_sides(*sides)
        assert tri.c_sum > 3.0 * np.sqrt(3.0) + 1e-9, name


# ---------------------------------------------------------------------
# G10c: wartości brzegowe i postać liniowa
# ---------------------------------------------------------------------

class TestClosedFormBoundaryValues:
    @pytest.mark.parametrize("name,sides", list(TRIANGLES.items()))
    def test_P_is_one_at_R_zero(self, name, sides):
        tri = TriangleGeometry.from_sides(*sides)
        assert P_of_R(tri, 0.0) == pytest.approx(1.0, abs=1e-12), name
        assert Q_of_R(tri, 0.0) == pytest.approx(0.0, abs=1e-12), name

    @pytest.mark.parametrize("name,sides", list(TRIANGLES.items()))
    def test_Lk_matches_2piR(self, name, sides):
        for R in (0.0, 0.1, 0.37):
            assert Lk_of_R(R) == pytest.approx(2 * np.pi * R)

    def test_R_beyond_R_max_raises(self):
        tri = TriangleGeometry.from_sides(3.0, 4.0, 5.0)
        with pytest.raises(ValueError):
            L0_of_R(tri, tri.R_max() + 0.01)

    def test_negative_R_raises(self):
        tri = TriangleGeometry.from_sides(3.0, 4.0, 5.0)
        with pytest.raises(ValueError):
            L0_of_R(tri, -0.1)


# ---------------------------------------------------------------------
# G10d: monotoniczność ścisła P(R) i odwracalność ("i odwrotnie")
# ---------------------------------------------------------------------

class TestMonotonicityAndInverse:
    @pytest.mark.parametrize("name,sides", list(TRIANGLES.items()))
    def test_P_strictly_decreasing(self, name, sides):
        tri = TriangleGeometry.from_sides(*sides)
        r_max = tri.R_max()
        Rs = np.linspace(0.0, r_max, 25)
        Ps = [P_of_R(tri, R) for R in Rs]
        diffs = np.diff(Ps)
        assert np.all(diffs < 0), f"{name}: P(R) powinno być ściśle malejące, dostano diffs={diffs}"

    @pytest.mark.parametrize("name,sides", list(TRIANGLES.items()))
    def test_R_of_P_inverts_P_of_R(self, name, sides):
        """'I odwrotnie' (G10d): dla R w [0,R_max], P_of_R -> R_of_P
        powinno odtworzyć oryginalne R."""
        tri = TriangleGeometry.from_sides(*sides)
        r_max = tri.R_max()
        for frac in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            R = frac * r_max
            P = P_of_R(tri, R)
            R_recovered = R_of_P(tri, P)
            assert R_recovered == pytest.approx(R, abs=1e-8), f"{name}, frac={frac}"

    def test_R_of_P_rejects_out_of_range(self):
        tri = TriangleGeometry.from_sides(3.0, 4.0, 5.0)
        with pytest.raises(ValueError):
            R_of_P(tri, 1.5)
        with pytest.raises(ValueError):
            R_of_P(tri, -0.1)

    def test_R_of_P_boundary_values(self):
        tri = TriangleGeometry.from_sides(3.0, 4.0, 5.0)
        assert R_of_P(tri, 1.0) == pytest.approx(0.0, abs=1e-9)
        assert R_of_P(tri, 0.0) == pytest.approx(tri.R_max(), abs=1e-9)


# ---------------------------------------------------------------------
# Niezależna weryfikacja numeryczna: skonstruowana polilinia ∂_R(Δ)
# vs zamknięty wzór L(R) — ten sam duch co T_S_empirical/T_S_predicted
# w weingarten.py.
# ---------------------------------------------------------------------

class TestBoundaryConstructionMatchesClosedForm:
    POINTS_345 = ([0.0, 0.0], [4.0, 0.0], [0.0, 3.0])  # bok 3(=|AC|... patrz from_points), 4, 5

    @pytest.mark.parametrize("R_frac", [0.0, 0.1, 0.5, 0.9, 0.999])
    def test_numeric_length_matches_closed_form(self, R_frac):
        A, B, C = self.POINTS_345
        tri = TriangleGeometry.from_points(A, B, C)
        R = R_frac * tri.R_max()
        result = verify_envelope_length(A, B, C, R, n_arc=300)
        assert result["relative_error"] < 2e-3, result

    def test_boundary_is_closed_polyline_with_expected_point_count(self):
        A, B, C = self.POINTS_345
        pts = rounded_triangle_boundary(A, B, C, R=0.3, n_arc=10)
        assert pts.shape[1] == 2
        assert len(pts) > 6  # co najmniej 3 odcinki + 3 łuki niezdegenerowane

    def test_rejects_3d_points(self):
        with pytest.raises(ValueError):
            rounded_triangle_boundary([0, 0, 0], [1, 0, 0], [0, 1, 0], R=0.1)

    def test_rejects_R_beyond_R_max(self):
        A, B, C = self.POINTS_345
        tri = TriangleGeometry.from_points(A, B, C)
        with pytest.raises(ValueError):
            rounded_triangle_boundary(A, B, C, R=tri.R_max() * 1.5)
