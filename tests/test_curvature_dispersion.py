"""
tests/test_curvature_dispersion.py

Testy Lambda_G (dyspersja sredniej krzywizny) - patrz PRE-REJESTRACJA w
naglowku `mean_curvature_dispersion()` w weingarten.py dla pelnego
uzasadnienia wzoru i odrzuconej alternatywy (parametr porzadku kierunkow
glownych - niemozliwy bez transportu rownoleglego miedzy przestrzeniami
stycznymi).

Kontrolki, dokladnie takie jak przedrejestrowane:
  + pozytywna (Lambda_G ~0): sfera i walec (krzywizna stala w czesci
    wewnetrznej).
  - negatywna (Lambda_G wyraznie wyzszy): plaszczyzna z losowym szumem
    z-wysokosci (niejednorodna krzywizna z konstrukcji).
Sprawdzany jest WZGLEDNY porzadek (szum > stala krzywizna), nie sztywny
prog liczbowy ustalony po fakcie - zgodnie z protokolem anty-numerologii.
"""
import numpy as np
import pytest

from timdr_geometry import (
    Mesh,
    make_plane_mesh,
    make_sphere_mesh,
    make_cylinder_mesh,
    mean_curvature_dispersion,
)


def test_sphere_has_near_zero_dispersion():
    mesh = make_sphere_mesh(n_lat=24, n_lon=24, radius=2.0)
    result = mean_curvature_dispersion(mesh)
    assert result.lambda_g < 0.1
    assert result.n_valid > 0


def test_cylinder_has_near_zero_dispersion():
    mesh = make_cylinder_mesh(n_theta=24, n_z=12, radius=1.5, height=3.0)
    result = mean_curvature_dispersion(mesh)
    assert result.lambda_g < 0.15


def test_flat_plane_has_zero_dispersion():
    """Przypadek zdegenerowany H=0 wszedzie (mean=0, std=0) - sprawdza
    ochrone przed 0/0 (eps w mianowniku), nie tylko 'niski' wynik."""
    mesh = make_plane_mesh(n=10, size=1.0)
    result = mean_curvature_dispersion(mesh)
    assert result.lambda_g < 1e-6
    assert not np.isnan(result.lambda_g)


def _noisy_plane_mesh(n: int, size: float, noise_std: float, seed: int) -> Mesh:
    """Siatka plaszczyzny z losowym szumem z-wysokosci na kazdym
    wierzcholku - z konstrukcji NIE ma stalej krzywizny (kontrola
    negatywna dla Lambda_G)."""
    rng = np.random.default_rng(seed)
    base = make_plane_mesh(n=n, size=size)
    verts = base.vertices.copy()
    verts[:, 2] += rng.normal(0.0, noise_std, size=verts.shape[0])
    return Mesh(verts, base.faces)


def test_noisy_plane_has_higher_dispersion_than_sphere_or_cylinder():
    """Kontrola negatywna: powierzchnia o niejednorodnej krzywiznie z
    konstrukcji powinna miec WYRAZNIE wyzszy Lambda_G niz powierzchnie o
    stalej krzywiznie (sfera/walec) - sprawdzamy wzgledny porzadek, nie
    konkretna wartosc progowa (protokol anty-numerologii, punkt 7)."""
    noisy = _noisy_plane_mesh(n=10, size=1.0, noise_std=0.3, seed=42)
    result_noisy = mean_curvature_dispersion(noisy)

    sphere = make_sphere_mesh(n_lat=24, n_lon=24, radius=2.0)
    result_sphere = mean_curvature_dispersion(sphere)

    cylinder = make_cylinder_mesh(n_theta=24, n_z=12, radius=1.5, height=3.0)
    result_cylinder = mean_curvature_dispersion(cylinder)

    assert result_noisy.lambda_g > result_sphere.lambda_g
    assert result_noisy.lambda_g > result_cylinder.lambda_g
    # Nie tylko wiekszy, ale wyraznie - inaczej to moglby byc szum
    # numeryczny, nie realny sygnal.
    assert result_noisy.lambda_g > 0.2


def test_dispersion_increases_with_noise_magnitude():
    """Dodatkowa kontrola monotonicznosci: wiecej szumu -> wiecej
    dyspersji, ten sam ziarno losowosci, rozne odchylenia standardowe."""
    low = _noisy_plane_mesh(n=10, size=1.0, noise_std=0.05, seed=7)
    high = _noisy_plane_mesh(n=10, size=1.0, noise_std=0.5, seed=7)
    result_low = mean_curvature_dispersion(low)
    result_high = mean_curvature_dispersion(high)
    assert result_high.lambda_g > result_low.lambda_g


def test_raises_on_mesh_with_no_faces_at_all():
    """Siatka bez zadnych trojkatow (same izolowane punkty) - zaden
    wierzcholek nie ma zadnego sasiedztwa w ogole (rings puste, normalne
    zerowe wszedzie) - powinno rzucic czytelny ValueError, nie cicho
    zwrocic NaN/0.

    UWAGA (poprawka po realnym wyniku testu): pierwsza wersja tego testu
    zakladala, ze pojedynczy trojkat (3 wierzcholki + 1 izolowany) da
    zero wazonych wierzcholkow - to bylo BLEDNE zalozenie, sprawdzone
    dopiero uruchomieniem testu (patrz test_minimum_two_neighbors_does_
    not_raise w test_geometry_weingarten.py - te same 3 wierzcholki MAJA
    dokladnie 2 sasiadow kazdy, wiec sa poprawnie liczone, tylko izolowany
    4. wierzcholek jest pomijany). Naprawione uzyciem siatki BEZ zadnych
    trojkatow w ogole, zamiast zgadywac dalej."""
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    faces = np.zeros((0, 3), dtype=int)
    mesh = Mesh(verts, faces)
    with pytest.raises(ValueError):
        mean_curvature_dispersion(mesh)
