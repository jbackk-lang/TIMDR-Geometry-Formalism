"""
timdr_geometry/envelope.py

Numeryczna implementacja Aksjomatu G10 — obwiednia zaokrąglona
`∂_R(Δ)` trójkąta i parametr redukcji/rozwinięcia `(P,Q)` — z
`GIA-TIMDR/docs/theory/Axioms_G_TIMDR_Geometry.md`.

G10a-e zostały wyprowadzone RĘCZNIE (nierówność Jensena, reguła
ilorazu, tożsamość stycznej-do-okręgu-wpisanego) i sprawdzone krok po
kroku na papierze — dokładnie ten sam status, co G8-G9 miały przed
`weingarten.py`. Ten moduł domyka to NUMERYCZNIE, tym samym wzorcem:
prawdziwy, uruchamialny kod plus testy na konkretnych trójkątach.

WAŻNE — poprawka błędu znalezionego w tej samej sesji. Pierwsza wersja
Aksjomatu G10e twierdziła, że złamanie symetrii trójkąta ściśle
ogranicza zasięg `Q` (`R_max(Δ)<r_in(Δ)` dla każdego trójkąta poza
równobocznym). To było MATEMATYCZNIE FAŁSZYWE — błąd wyszedł na jaw
dopiero przy liczeniu konkretnego przykładu liczbowego (trójkąt 3-4-5)
do innego dokumentu, nie przy przeglądzie samego aksjomatu. Poprawiona
wersja (obecna w Axioms_G_TIMDR_Geometry.md): ze standardowej tożsamości
`cot(θᵢ/2)=(s-aᵢ)/r_in` wynika `R_max(Δ)=r_in(Δ)` DOKŁADNIE dla
KAŻDEGO trójkąta, więc `Q=1` jest osiągalne zawsze (w granicy
`R→R_max`), nie tylko dla trójkąta równobocznego. Testy w
`tests/test_envelope.py` weryfikują numerycznie WŁAŚNIE tę poprawioną
tożsamość (na trójkącie ostrym, rozwartym, prawie zdegenerowanym) —
`R_max` i `r_in` są tu policzone DWIEMA NIEZALEŻNYMI ścieżkami (suma
kotangensów połówek kątów vs wzór Herona), żeby test nie był kołowy.
To, co POZOSTAJE prawdą: przy ustalonym obwodzie trójkąt równoboczny ma
NAJWIĘKSZY `r_in` (nierówność Jensena, G10d), więc symetria wpływa na
BEZWZGLĘDNY promień `R`, przy którym `Q→1` następuje, nie na to, czy w
ogóle następuje.

Co ten moduł liczy (G10a-d):
    1. Geometria trójkąta z 3 punktów lub z długości boków
       -> TriangleGeometry.from_points() / .from_sides()
    2. L0(R), Lk(R), L(R), P(R), Q(R)   -> funkcje modułowe
    3. R_max(Δ) (dwiema niezależnymi ścieżkami, patrz wyżej)
    4. Odwrotność "i odwrotnie": zadany P -> R -> obwiednia
       -> R_of_P()
    5. Sama obwiednia jako polilinia (odcinki+łuki), do NIEZALEŻNEJ
       numerycznej weryfikacji L0/Lk przez zmierzenie długości
       skonstruowanej krzywej -> rounded_triangle_boundary(),
       verify_envelope_length()

UWAGA O WALIDACJI: ten moduł implementuje i numerycznie weryfikuje
G10a-e na trójkątach SYNTETYCZNYCH (podanych explicite, nie zmierzonych)
— to jest test poprawności matematyki/implementacji, nie walidacja
empiryczna na rzeczywistych danych geometrycznych (ten sam status, co
`weingarten.py` ma dla G8-G9 — patrz zastrzeżenie tam i w Aksjomacie
G7c). Status gałęzi G pozostaje "koncepcyjna" (G7a).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "TriangleGeometry",
    "L0_of_R",
    "Lk_of_R",
    "L_of_R",
    "P_of_R",
    "Q_of_R",
    "R_of_P",
    "rounded_triangle_boundary",
    "boundary_length_numeric",
    "verify_envelope_length",
]


# ---------------------------------------------------------------------
# Geometria trójkąta (Aksjomat G2) — boki, kąty, s, obszar, r_in, c(Δ)
# ---------------------------------------------------------------------

@dataclass
class TriangleGeometry:
    """Metryczne dane trójkąta Δ=(A,B,C) potrzebne do G10a-e.

    Konwencja: `sides=(a,b,c)` gdzie `a=|BC|` (przeciwlegly A),
    `b=|CA|` (przeciwlegly B), `c=|AB|` (przeciwlegly C) — standardowa
    konwencja trygonometryczna. `angles=(alpha,beta,gamma)` w
    radianach, w tej samej kolejności (alpha przy A, itd.).
    """

    sides: np.ndarray        # (a, b, c)
    angles: np.ndarray       # (alpha, beta, gamma), radiany
    P0: float                # obwód = a+b+c
    s: float                 # połobwód
    area: float               # pole (wzór Herona)
    r_in: float               # promień okręgu wpisanego = area/s
    c_sum: float              # c(Delta) = sum cot(theta_i/2)

    @staticmethod
    def from_sides(a: float, b: float, c: float) -> "TriangleGeometry":
        """Buduje geometrię z trzech długości boków. Rzuca ValueError,
        jeśli nie spełniają ściśle nierówności trójkąta (zdegenerowany
        albo niemożliwy trójkąt) — nigdy nie zwraca cichych NaN-ów."""
        a, b, c = float(a), float(b), float(c)
        if a <= 0 or b <= 0 or c <= 0:
            raise ValueError(f"długości boków muszą być dodatnie, dostano ({a},{b},{c})")
        if not (a + b > c and b + c > a and c + a > b):
            raise ValueError(
                f"boki ({a},{b},{c}) nie spełniają ściśle nierówności trójkąta "
                "(zdegenerowany lub niemożliwy trójkąt)"
            )

        # Kąty z twierdzenia cosinusów — ścieżka NIEZALEŻNA od r_in,
        # celowo (patrz uwaga w docstringu modułu o unikaniu kołowości
        # przy testowaniu R_max==r_in).
        alpha = np.arccos(np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0))
        beta = np.arccos(np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1.0, 1.0))
        gamma = np.pi - alpha - beta

        P0 = a + b + c
        s = P0 / 2.0
        area = float(np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0)))
        if area <= 0:
            raise ValueError(f"boki ({a},{b},{c}) dają zdegenerowany (zerowe pole) trójkąt")
        r_in = area / s

        c_sum = float(
            _cot_half(alpha) + _cot_half(beta) + _cot_half(gamma)
        )

        return TriangleGeometry(
            sides=np.array([a, b, c]),
            angles=np.array([alpha, beta, gamma]),
            P0=P0, s=s, area=area, r_in=r_in, c_sum=c_sum,
        )

    @staticmethod
    def from_points(A: Sequence[float], B: Sequence[float], C: Sequence[float]) -> "TriangleGeometry":
        """Buduje geometrię z 3 punktów (dowolny wymiar — 2D lub 3D,
        bo G10a-d zależą tylko od długości boków/kątów, nie od
        osadzenia). Dla samej obwiedni jako krzywej płaskiej patrz
        `rounded_triangle_boundary()`, która wymaga punktów 2D."""
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        C = np.asarray(C, dtype=float)
        a = float(np.linalg.norm(B - C))  # przeciwlegly A
        b = float(np.linalg.norm(C - A))  # przeciwlegly B
        c = float(np.linalg.norm(A - B))  # przeciwlegly C
        return TriangleGeometry.from_sides(a, b, c)

    def R_max(self) -> float:
        """R_max(Δ) = min_{i,j} a_ij/(cot(θᵢ/2)+cot(θⱼ/2)) — Aksjomat
        G10a, policzone WPROST z definicji (nie jako skrót do r_in),
        żeby test `R_max ≈ r_in` (poprawka G10e) był niekołowy."""
        a, b, c = self.sides
        alpha, beta, gamma = self.angles
        cot_a, cot_b, cot_g = _cot_half(alpha), _cot_half(beta), _cot_half(gamma)
        candidates = [
            a / (cot_b + cot_g),   # bok a leży między wierzchołkami B,C
            b / (cot_a + cot_g),   # bok b leży między wierzchołkami A,C
            c / (cot_a + cot_b),   # bok c leży między wierzchołkami A,B
        ]
        return float(min(candidates))

    def is_equilateral(self, tol: float = 1e-9) -> bool:
        a, b, c = self.sides
        return bool(np.isclose(a, b, atol=tol) and np.isclose(b, c, atol=tol))


def _cot_half(theta: float) -> float:
    """cot(theta/2) = (1+cos theta)/sin theta — G10a/G10e."""
    return (1.0 + np.cos(theta)) / np.sin(theta)


# ---------------------------------------------------------------------
# L0(R), Lk(R), L(R), P(R), Q(R) — Aksjomat G10b-c, postać jawna/liniowa
# ---------------------------------------------------------------------

def Lk_of_R(R: float) -> float:
    """Lk(R) = 2*pi*R dokładnie — G10c, z sumy kątów zewnętrznych
    dowolnego wypukłego wielokąta. Niezależne od kształtu Δ."""
    if R < 0:
        raise ValueError(f"R musi być >=0, dostano {R}")
    return 2.0 * np.pi * R


def L0_of_R(tri: TriangleGeometry, R: float) -> float:
    """L0(R) = P0 - 2*R*c(Δ) — G10c. Rzuca ValueError, jeśli R wychodzi
    poza dziedzinę konstrukcji [0, R_max(Δ)) (G10a: obwiednia dobrze
    określona tylko tam — poza nią boki byłyby skrócone do długości
    ujemnej, czyli łuki by się przecinały)."""
    if R < 0:
        raise ValueError(f"R musi być >=0, dostano {R}")
    r_max = tri.R_max()
    if R > r_max + 1e-9:
        raise ValueError(
            f"R={R} przekracza R_max(Δ)={r_max:.6f} — konstrukcja ∂_R(Δ) "
            "nie jest tam dobrze określona (G10a)"
        )
    return tri.P0 - 2.0 * R * tri.c_sum


def L_of_R(tri: TriangleGeometry, R: float) -> float:
    """L(R) = L0(R) + Lk(R) — G10b."""
    return L0_of_R(tri, R) + Lk_of_R(R)


def P_of_R(tri: TriangleGeometry, R: float) -> float:
    """P(R) = L0(R)/L(R) — G10b, "redukcja": obwiednia -> parametr."""
    L = L_of_R(tri, R)
    return L0_of_R(tri, R) / L


def Q_of_R(tri: TriangleGeometry, R: float) -> float:
    """Q(R) = Lk(R)/L(R) = 1-P(R) — G10b."""
    return 1.0 - P_of_R(tri, R)


def R_of_P(tri: TriangleGeometry, P: float) -> float:
    """Odwrotność P(R) w postaci zamkniętej — Aksjomat G10d, kierunek
    "rozwinięcie" ("i odwrotnie"): zadany P -> R -> ∂_R(Δ).

    Wyprowadzenie: L0,Lk są afiniczne w R (G10c), więc
    P(R)=(P0-2cR)/(P0-2kR) z k:=c-pi jest funkcją Möbiusa (homograficzną)
    w R — odwracalną w zamkniętej postaci:
        R(P) = P0*(1-P) / (2*(c - P*k))
    Mianownik `2*(c-P*k) >= 2*(c-k) = 2*pi > 0` dla P<=1 zawsze (bo
    k=c-pi<c z G10d), więc nigdy nie dzieli przez zero na [0,1].
    Sprawdzone: R(1)=0 (czysty trójkąt), R(0)=P0/(2c)=r_in=R_max
    (poprawka G10e — Q=1 osiągalne zawsze, nie tylko dla równobocznego).
    """
    if not (0.0 <= P <= 1.0):
        raise ValueError(f"P musi być w [0,1], dostano {P}")
    c = tri.c_sum
    k = c - np.pi
    denom = 2.0 * (c - P * k)
    R = tri.P0 * (1.0 - P) / denom
    return float(R)


# ---------------------------------------------------------------------
# Obwiednia jako krzywa płaska (odcinki+łuki) — niezależna weryfikacja
# numeryczna L0(R)/Lk(R) przez zmierzenie skonstruowanej geometrii
# ---------------------------------------------------------------------

def rounded_triangle_boundary(
    A: Sequence[float], B: Sequence[float], C: Sequence[float],
    R: float, n_arc: int = 64,
) -> np.ndarray:
    """Buduje ∂_R(Δ) (Aksjomat G10a) jako polilinię (N,2) — 3 odcinki
    proste + 3 łuki próbkowane numerycznie, w płaszczyźnie trójkąta.
    Wymaga punktów 2D (obwiednia jest krzywą płaską — patrz docstring
    modułu). Nie zakłada orientacji (CW/CCW) — kąt wymiatany każdego
    łuku jest mierzony wprost z geometrii (atan2), nie zgadywany.

    Używane do NIEZALEŻNEJ weryfikacji: sumowanie długości tej
    polilinii powinno zbiegać (przy rosnącym n_arc) do L0(R)+Lk(R) z
    zamkniętych wzorów G10c — patrz verify_envelope_length()."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    if A.shape != (2,) or B.shape != (2,) or C.shape != (2,):
        raise ValueError("rounded_triangle_boundary wymaga punktów 2D (obwiednia jest krzywą płaską)")
    if R < 0:
        raise ValueError(f"R musi być >=0, dostano {R}")

    tri = TriangleGeometry.from_points(A, B, C)
    r_max = tri.R_max()
    if R > r_max + 1e-9:
        raise ValueError(f"R={R} przekracza R_max(Δ)={r_max:.6f}")

    verts = [A, B, C]
    angles = tri.angles  # (alpha,beta,gamma) odpowiadajace A,B,C

    def tangent_len(theta: float) -> float:
        return R * _cot_half(theta)

    # Dla kazdego wierzcholka: punkty styczne na obu sasiednich bokach
    # + srodek luku (wzdluz dwusiecznej, w odleglosci R/sin(theta/2)).
    in_tan = [None, None, None]   # punkt styczny na boku "wchodzacym" (prev->this)
    out_tan = [None, None, None]  # punkt styczny na boku "wychodzacym" (this->next)
    centers = [None, None, None]
    for i in range(3):
        P = verts[i]
        P_next = verts[(i + 1) % 3]
        P_prev = verts[(i - 1) % 3]
        theta = angles[i]
        t = tangent_len(theta)

        dir_next = (P_next - P) / np.linalg.norm(P_next - P)
        dir_prev = (P_prev - P) / np.linalg.norm(P_prev - P)

        out_tan[i] = P + dir_next * t          # styczny na boku P->P_next
        in_tan[i] = P + dir_prev * t            # styczny na boku P_prev->P (od strony P)

        bis = dir_next + dir_prev
        bis_norm = np.linalg.norm(bis)
        if bis_norm < 1e-12:
            raise ValueError(f"wierzchołek {i}: zdegenerowana dwusieczna (kąt ~0 lub ~pi)")
        bis = bis / bis_norm
        dist = R / np.sin(theta / 2.0) if R > 0 else 0.0
        centers[i] = P + bis * dist

    def arc_points(center, p_from, p_to, n):
        if R == 0.0:
            return np.array(p_to).reshape(1, 2)
        u = (np.asarray(p_from) - center)
        v = (np.asarray(p_to) - center)
        ang_u = np.arctan2(u[1], u[0])
        ang_v = np.arctan2(v[1], v[0])
        dtheta = ang_v - ang_u
        # znormalizuj do (-pi, pi], zeby zawsze isc "krotsza" droga —
        # z konstrukcji G10a kat wymiatany ma dokladnie magnitude
        # pi-theta_i < pi, wiec ten wybor jest jednoznaczny i poprawny.
        while dtheta > np.pi:
            dtheta -= 2 * np.pi
        while dtheta <= -np.pi:
            dtheta += 2 * np.pi
        ts = np.linspace(0.0, dtheta, n)
        pts = np.stack([center[0] + R * np.cos(ang_u + ts), center[1] + R * np.sin(ang_u + ts)], axis=1)
        return pts

    poly = []
    for i in range(3):
        j = (i + 1) % 3
        # odcinek prosty: od out_tan wierzcholka i do in_tan wierzcholka j
        poly.append(out_tan[i].reshape(1, 2))
        poly.append(in_tan[j].reshape(1, 2))
        # luk w wierzcholku j: od in_tan[j] do out_tan[j]
        poly.append(arc_points(centers[j], in_tan[j], out_tan[j], n_arc))
    return np.concatenate(poly, axis=0)


def boundary_length_numeric(points: np.ndarray) -> float:
    """Długość zamkniętej polilinii (N,2) — suma odległości kolejnych
    punktów plus odcinek zamykający (ostatni->pierwszy)."""
    points = np.asarray(points, dtype=float)
    diffs = np.diff(points, axis=0, append=points[:1])
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def verify_envelope_length(
    A: Sequence[float], B: Sequence[float], C: Sequence[float],
    R: float, n_arc: int = 200,
) -> dict:
    """Porównuje L(R) z zamkniętego wzoru G10c z długością zmierzoną
    numerycznie na skonstruowanej polilinii ∂_R(Δ) — niezależna
    weryfikacja, analogiczna do T_S_empirical vs T_S_predicted w
    weingarten.py. Zwraca dict z obiema wartościami i błędem względnym."""
    tri = TriangleGeometry.from_points(A, B, C)
    closed_form = L_of_R(tri, R)
    poly = rounded_triangle_boundary(A, B, C, R, n_arc=n_arc)
    numeric = boundary_length_numeric(poly)
    rel_err = abs(numeric - closed_form) / closed_form if closed_form > 0 else float("nan")
    return dict(closed_form=closed_form, numeric=numeric, relative_error=rel_err)
