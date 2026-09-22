# tests/test_spectral_family.py
"""Testy jednostkowe "czy operator robi to co mowi" dla
timdr_geometry/spectral_family.py (rdzen matematyczny wydzielony z
GIA-TIMDR/core/chrono_membrane_bridge.py -- patrz tamtejszy
docstring modulu oraz GIA-TIMDR/docs/geometry/
PREREG_CHRONO_MEMBRANE_BEARING_v0.1.md po kontekst historyczny).
Kontrole syntetyczne (bramka pozytywna/negatywna na pelnej siatce) sa
w timdr_geometry/spectral_family.py (uruchamiane jako skrypt). Te
testy sprawdzaja WYLACZNIE elementarne wlasciwosci funkcji budujacych
(macierz korelacji, widmo, metryki), nie powtarzaja calej bramki
kontrolnej.
"""
import numpy as np
import pytest

from timdr_geometry.spectral_family import (
    channel_correlation_matrix,
    spectrum_from_correlation,
    spectral_concentration,
    participation_ratio,
    membrane_window_metrics,
    make_growing_shared,
    make_independent_noise,
    make_constant_shared,
)


def test_correlation_matrix_identical_channels_is_all_ones():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    C = channel_correlation_matrix([x, x, x])
    assert C.shape == (3, 3)
    assert np.allclose(C, 1.0, atol=1e-8)


def test_correlation_matrix_diagonal_is_one():
    rng = np.random.default_rng(1)
    chans = [rng.normal(0, 1, 300) for _ in range(3)]
    C = channel_correlation_matrix(chans)
    assert np.allclose(np.diag(C), 1.0)


def test_spectrum_identical_channels_fully_concentrated():
    x = np.linspace(0, 1, 100) + np.sin(np.linspace(0, 20, 100))
    C = channel_correlation_matrix([x, x])
    eig = spectrum_from_correlation(C)
    assert eig[0] == pytest.approx(2.0, abs=1e-6)
    assert eig[1] == pytest.approx(0.0, abs=1e-6)
    assert spectral_concentration(eig) == pytest.approx(1.0, abs=1e-6)
    assert participation_ratio(eig) == pytest.approx(1.0, abs=1e-6)


def test_spectrum_nonnegative_and_descending():
    rng = np.random.default_rng(2)
    chans = [rng.normal(0, 1, 150) for _ in range(4)]
    C = channel_correlation_matrix(chans)
    eig = spectrum_from_correlation(C)
    assert np.all(eig >= -1e-9)
    assert np.all(np.diff(eig) <= 1e-9)  # malejaco


def test_participation_ratio_bounds():
    rng = np.random.default_rng(3)
    chans = [rng.normal(0, 1, 500) for _ in range(3)]
    C = channel_correlation_matrix(chans)
    eig = spectrum_from_correlation(C)
    pr = participation_ratio(eig)
    assert 1.0 <= pr <= 3.0 + 1e-6


def test_membrane_window_metrics_keys_present():
    rng = np.random.default_rng(4)
    chans = [rng.normal(0, 1, 256) for _ in range(2)]
    m = membrane_window_metrics(chans)
    assert set(m.keys()) == {"spectral_concentration", "participation_ratio", "membrane_spectral_ratio"}
    assert 0.5 <= m["spectral_concentration"] <= 1.0
    assert 1.0 <= m["participation_ratio"] <= 2.0 + 1e-6


def test_membrane_ratio_nan_for_too_short_window():
    rng = np.random.default_rng(5)
    chans = [rng.normal(0, 1, 8) for _ in range(3)]  # zbyt krotkie wzgledem N=3
    m = membrane_window_metrics(chans)
    assert np.isnan(m["membrane_spectral_ratio"])


def test_generators_shapes():
    for gen in (make_growing_shared, make_independent_noise, make_constant_shared):
        chans = gen(3, 128, 0)
        assert len(chans) == 3
        assert all(len(c) == 128 for c in chans)


def test_growing_shared_increases_end_concentration_typically():
    # sanity, nie formalny test statystyczny (ten jest w run_membrane_controls) --
    # sprawdza tylko, ze mechanizm dziala w oczekiwanym kierunku na jednej realizacji
    # o duzym window_size (mniej szumu w oszacowaniu).
    chans = make_growing_shared(3, 1000, seed=42)
    m = membrane_window_metrics(chans)
    assert m["membrane_spectral_ratio"] > 1.0
