"""
timdr_geometry/weingarten.py

Dyskretny operator kształtu (Weingarten) na trójkątnej siatce 3D —
numeryczna implementacja Aksjomatów G8-G9 z
`GIA-TIMDR/docs/theory/Axioms_G_TIMDR_Geometry.md`.

G8 sformalizował skręt powierzchniowy T_S jako operator z jawną domeną,
przeciwdziedzią, ciągłością i stabilnością. G9 podał jawną postać
związku ze krzywizną przez operator Weingartena — ANALITYCZNIE
(tożsamość różniczkowo-geometryczna, prawdziwa w granicy Δp→0, plus
standardowa dyskretna aproksymacja różnicą skończoną). Ten moduł
domyka to NUMERYCZNIE: prawdziwy, uruchamialny kod, który liczy
dyskretny operator kształtu na konkretnej siatce.

Konwencja znaku (WAŻNE — różni się między podręcznikami geometrii
różniczkowej): S_p(v) := -D_v n(p), zgodnie z Aksjomatem G9a. Dla
powierzchni wypukłej z normalną skierowaną na zewnątrz (np. sfera) to
daje UJEMNE wartości własne. Ponieważ jedyne użycie tego operatora w
Aksjomacie G8d (ograniczenie Lipschitza κ_max = max(|κ1|,|κ2|)) zależy
wyłącznie od WARTOŚCI BEZWZGLĘDNEJ krzywizn głównych, ten moduł i jego
testy sprawdzają magnitudy, nie znak — znak jest udokumentowaną
konwencją, nie błędem, jeśli wychodzi ujemny dla wypukłej powierzchni.

Krok po kroku (algorytm, zgodny z opisem w GIA-TIMDR/docs/theory):
    1. Normalne wierzchołkowe    -> vertex_normals()
    2. Różnice normalnych Δn     -> wewnątrz discrete_shape_operator()
    3. Rzut na styczną           -> project_tangent()
    4. Liniowa regresja (fit)    -> discrete_shape_operator()
    5. Wartości własne (eigen)   -> ShapeOperatorResult.principal_curvatures

UWAGA O WALIDACJI: ten moduł implementuje G9 numerycznie, ale sam przez
to NIE zamyka Aksjomatu G7c w całości — punkty (2) formalna przestrzeń
powierzchni, (3) testy empiryczne NA RZECZYWISTYCH danych geometrycznych
(nie syntetycznych), (4) niezależna walidacja, pozostają otwarte.
Testy w tym repo (tests/test_geometry_weingarten.py) sprawdzają
poprawność implementacji na trzech przypadkach ZNANYCH ANALITYCZNIE
(płaszczyzna, sfera, walec) — to jest test poprawności kodu, nie
walidacja empiryczna w sensie G7c(3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "Mesh",
    "ShapeOperatorResult",
    "vertex_normals",
    "one_ring",
    "tangent_basis",
    "project_tangent",
    "discrete_shape_operator",
    "T_S_empirical",
    "T_S_predicted",
    "kappa_max",
    "mean_curvature",
    "gaussian_curvature",
    "make_plane_mesh",
    "make_sphere_mesh",
    "make_cylinder_mesh",
]


# ---------------------------------------------------------------------
# Obiekt podstawowy: siatka trójkątna (Aksjomat G1 — wersja dyskretna)
# ---------------------------------------------------------------------

@dataclass
class Mesh:
    """Trójkątna siatka powierzchni S⊂R³ (dyskretny odpowiednik
    dopuszczalnej powierzchni z Aksjomatu G1).

    vertices: (N,3) float — wierzchołki p_i.
    faces: (M,3) int — indeksy wierzchołków trójkątów. Orientacja
        (kolejność wierzchołków) decyduje o znaku n(p) — patrz uwaga o
        konwencji znaku na górze pliku.
    """

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float)
        self.faces = np.asarray(self.faces, dtype=int)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices musi mieć kształt (N,3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces musi mieć kształt (M,3)")

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]


def _face_normals_unnormalized(mesh: Mesh) -> np.ndarray:
    """Normalne trójkątów, NIEznormalizowane — długość = 2*pole trójkąta,
    co daje darmowe ważenie polem przy sumowaniu w vertex_normals()."""
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]
    return np.cross(v1 - v0, v2 - v0)


def vertex_normals(mesh: Mesh) -> np.ndarray:
    """Krok 1: normalne wierzchołkowe n(p) — suma normalnych sąsiednich
    trójkątów ważona polem (bo używamy nieznormalizowanych normalnych
    trójkątów, których długość = 2*pole), potem znormalizowana.

    Zgodnie z Aksjomatem G1b (n określone p.w.): wierzchołek nie należący
    do żadnego trójkąta dostaje wektor zerowy — jawny sygnał
    "nieokreślone", nie cichy błąd numeryczny (dzielenie przez zero).
    """
    face_normals = _face_normals_unnormalized(mesh)
    acc = np.zeros_like(mesh.vertices)
    for f_idx, face in enumerate(mesh.faces):
        acc[face] += face_normals[f_idx]
    norms = np.linalg.norm(acc, axis=1)
    out = np.zeros_like(acc)
    nonzero = norms > 0
    out[nonzero] = acc[nonzero] / norms[nonzero, None]
    return out


def one_ring(mesh: Mesh) -> list:
    """Sąsiedztwo 1-ring: dla każdego wierzchołka, zbiór indeksów
    wierzchołków połączonych z nim krawędzią (przez wspólny trójkąt)."""
    rings: list = [set() for _ in range(mesh.n_vertices)]
    for a, b, c in mesh.faces:
        rings[a].update((int(b), int(c)))
        rings[b].update((int(a), int(c)))
        rings[c].update((int(a), int(b)))
    return rings


def tangent_basis(n_p: np.ndarray) -> tuple:
    """Dwa ortonormalne wektory rozpinające płaszczyznę styczną
    (prostopadłą do n_p). Wybór bazy jest arbitralny (dowolny obrót w
    płaszczyźnie stycznej) — nie wpływa na wartości własne S_p (są
    niezmiennicze względem zmiany bazy ortonormalnej), tylko na jego
    zapis macierzowy pośredni."""
    n_p = n_p / np.linalg.norm(n_p)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n_p[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = helper - np.dot(helper, n_p) * n_p
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n_p, e1)
    return e1, e2


def project_tangent(v: np.ndarray, n_p: np.ndarray) -> np.ndarray:
    """Krok 3: rzut wektora v na płaszczyznę styczną w punkcie z
    normalną n_p (zakładana jednostkowa): t = v - <v,n_p> n_p."""
    return v - np.dot(v, n_p) * n_p


# ---------------------------------------------------------------------
# Dyskretny operator kształtu S_p (Aksjomat G9b) i jego dopasowanie MNK
# ---------------------------------------------------------------------

@dataclass
class ShapeOperatorResult:
    """Dopasowany dyskretny operator kształtu S_p w jednym wierzchołku."""

    point_idx: int
    n_p: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    matrix_2d: np.ndarray               # 2x2, symetryzowane, w bazie (e1,e2)
    principal_curvatures: np.ndarray    # (kappa1, kappa2), malejąco po wartości
    principal_directions_2d: np.ndarray  # kolumny = kierunki własne w bazie (e1,e2)
    n_neighbors: int

    def apply(self, v: np.ndarray) -> np.ndarray:
        """S_p(v) w R³: rzutuje v na styczną, aplikuje macierz 2x2,
        zwraca wynik jako wektor 3D leżący w płaszczyźnie stycznej."""
        t = project_tangent(np.asarray(v, dtype=float), self.n_p)
        coords = np.array([np.dot(t, self.e1), np.dot(t, self.e2)])
        out2d = self.matrix_2d @ coords
        return out2d[0] * self.e1 + out2d[1] * self.e2


def discrete_shape_operator(
    mesh: Mesh,
    normals: np.ndarray,
    point_idx: int,
    rings: Optional[list] = None,
) -> ShapeOperatorResult:
    """Krok 4 ("fit"): dyskretny operator kształtu S_p w wierzchołku p,
    metodą najmniejszych kwadratów na 1-ringu — Aksjomat G9b.

    Dla każdego sąsiada q w 1-ringu p: Δp=q-p, Δn=n(q)-n(p), oba
    zrzutowane na płaszczyznę styczną w p i wyrażone w lokalnej bazie
    ortonormalnej (e1,e2). Szukana macierz 2x2 A minimalizuje
    Σᵢ‖A tᵢ − Δnᵢ‖² (najmniejsze kwadraty), potem symetryzowana
    (A+Aᵀ)/2 — prawdziwy operator kształtu jest samosprzężony;
    asymetria czystego dopasowania MNK bierze się z nierównomiernego
    próbkowania 1-ringu, nie z realnej asymetrii operatora (standardowa
    poprawka w dyskretnej geometrii różniczkowej).

    Znak: A jest dopasowane wprost do Δn = n(q)-n(p) (BEZ dodatkowego
    minusa) — czyli A ≈ D_v n(p). Żeby zgadzało się to z konwencją
    S_p = -D_v n(p) z Aksjomatu G9a, wynik jest jawnie negowany przed
    zwróceniem. Patrz uwaga o konwencji znaku na górze pliku.

    Rzuca ValueError, jeśli wierzchołek ma <2 sąsiadów (izolowany albo
    na skraju siatki bez pełnego 1-ringu) albo jeśli rzuty styczne
    sąsiadów są zdegenerowane (rząd <2, MNK niedookreślone) — nigdy nie
    zwraca cichej macierzy zerowej w tych przypadkach.
    """
    if rings is None:
        rings = one_ring(mesh)
    neighbors = sorted(rings[point_idx])
    if len(neighbors) < 2:
        raise ValueError(
            f"wierzchołek {point_idx} ma tylko {len(neighbors)} sąsiadów "
            "— dyskretny operator kształtu wymaga >=2 do dopasowania MNK"
        )

    p = mesh.vertices[point_idx]
    n_p = normals[point_idx]
    if np.linalg.norm(n_p) == 0:
        raise ValueError(f"n(p) nieokreślone (wektor zerowy) dla wierzchołka {point_idx}")
    n_p = n_p / np.linalg.norm(n_p)
    e1, e2 = tangent_basis(n_p)

    T_rows = []
    N_rows = []
    for q_idx in neighbors:
        q = mesh.vertices[q_idx]
        n_q = normals[q_idx]
        if np.linalg.norm(n_q) == 0:
            continue  # sąsiad bez określonej normalnej — pomiń, nie licz
        delta_p = q - p
        delta_n = n_q - n_p
        t = project_tangent(delta_p, n_p)
        dn_t = project_tangent(delta_n, n_p)
        T_rows.append([np.dot(t, e1), np.dot(t, e2)])
        N_rows.append([np.dot(dn_t, e1), np.dot(dn_t, e2)])

    if len(T_rows) < 2:
        raise ValueError(
            f"wierzchołek {point_idx}: po odfiltrowaniu sąsiadów bez "
            "określonej normalnej zostało <2 użytecznych sąsiadów"
        )

    T = np.array(T_rows)
    N = np.array(N_rows)

    if np.linalg.matrix_rank(T) < 2:
        raise ValueError(
            f"wierzchołek {point_idx}: rzuty styczne sąsiadów są "
            "zdegenerowane (rząd <2) — MNK niedookreślone lokalnie"
        )

    X, *_ = np.linalg.lstsq(T, N, rcond=None)   # T @ X ≈ N,  X: (2,2)
    A = X.T                                     # konwencja: A @ t ≈ Δn  (≈ D_v n)
    A = -A                                      # S_p := -D_v n  (Aksjomat G9a)
    A_sym = 0.5 * (A + A.T)

    eigvals, eigvecs = np.linalg.eigh(A_sym)     # rosnąco
    order = np.argsort(eigvals)[::-1]            # przestaw malejąco
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    return ShapeOperatorResult(
        point_idx=point_idx,
        n_p=n_p,
        e1=e1,
        e2=e2,
        matrix_2d=A_sym,
        principal_curvatures=eigvals,
        principal_directions_2d=eigvecs,
        n_neighbors=len(T_rows),
    )


def kappa_max(op: ShapeOperatorResult) -> float:
    """κ_max(p) = max(|κ1|,|κ2|) — dokładnie stała Lipschitza z
    Aksjomatu G8d."""
    return float(np.max(np.abs(op.principal_curvatures)))


def mean_curvature(op: ShapeOperatorResult) -> float:
    """H = (κ1+κ2)/2 — ślad/2 operatora kształtu (Aksjomat G9a)."""
    return float(np.mean(op.principal_curvatures))


def gaussian_curvature(op: ShapeOperatorResult) -> float:
    """K = κ1·κ2 — wyznacznik operatora kształtu (Aksjomat G9a)."""
    return float(np.prod(op.principal_curvatures))


# ---------------------------------------------------------------------
# T_S: bezpośredni pomiar (Aksjomat G3) vs przewidywanie z S_p (G9c)
# ---------------------------------------------------------------------

def T_S_empirical(mesh: Mesh, normals: np.ndarray, p_idx: int, q_idx: int) -> float:
    """T_S(p) dla konkretnej krawędzi p→q na siatce, wprost z Aksjomatu
    G3: T_S = ‖n(q) − n(p)‖. Nie używa operatora kształtu w ogóle —
    to jest "prawda gruntowa" do porównania z T_S_predicted()."""
    return float(np.linalg.norm(normals[q_idx] - normals[p_idx]))


def T_S_predicted(op: ShapeOperatorResult, delta_p: np.ndarray) -> float:
    """F(W_S)(p,Δp) z Aksjomatu G9c: ‖Δp‖·‖S_p(Δ̂p)‖ — przybliżenie
    T_S(p) do pierwszego rzędu, WYŁĄCZNIE z operatora kształtu, bez
    bezpośredniego odczytu n(q). Zgodność z T_S_empirical do rzędu
    O(‖Δp‖²) jest dokładnie tym, co Aksjomat G9c twierdzi — testowana w
    tests/test_geometry_weingarten.py."""
    delta_p = np.asarray(delta_p, dtype=float)
    norm_dp = float(np.linalg.norm(delta_p))
    if norm_dp == 0.0:
        return 0.0
    direction = delta_p / norm_dp
    return norm_dp * float(np.linalg.norm(op.apply(direction)))


# ---------------------------------------------------------------------
# Generatory siatek testowych (przypadki znane analitycznie)
# ---------------------------------------------------------------------

def make_plane_mesh(n: int = 10, size: float = 1.0) -> Mesh:
    """Płaska siatka n×n na kwadracie [0,size]², z=0. Oczekiwane:
    S_p≡0 wszędzie, T_S≡0 (Test 2)."""
    xs = np.linspace(0.0, size, n)
    ys = np.linspace(0.0, size, n)
    verts = np.array([[x, y, 0.0] for y in ys for x in xs])
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = j * n + (i + 1)
            c = (j + 1) * n + i
            d = (j + 1) * n + (i + 1)
            faces.append([a, b, d])
            faces.append([a, d, c])
    return Mesh(verts, np.array(faces))


def make_sphere_mesh(n_lat: int = 20, n_lon: int = 20, radius: float = 1.0) -> Mesh:
    """Siatka sfery typu lat-long (bieguny celowo pominięte jako
    osobne, zdegenerowane wierzchołki — znana, zaakceptowana wada
    siatek lat-long; wiersze blisko biegunów/brzegu siatki należy
    pominąć w testach, patrz margin w testach). Oczekiwane: obie
    krzywizny główne o wartości bezwzględnej 1/radius wszędzie
    (Test 1)."""
    lats = np.linspace(np.pi / n_lat, np.pi - np.pi / n_lat, n_lat)
    lons = np.linspace(0.0, 2 * np.pi, n_lon, endpoint=False)
    verts = []
    for lat in lats:
        for lon in lons:
            verts.append(
                [
                    radius * np.sin(lat) * np.cos(lon),
                    radius * np.sin(lat) * np.sin(lon),
                    radius * np.cos(lat),
                ]
            )
    verts = np.array(verts)
    faces = []
    for j in range(n_lat - 1):
        for i in range(n_lon):
            a = j * n_lon + i
            b = j * n_lon + (i + 1) % n_lon
            c = (j + 1) * n_lon + i
            d = (j + 1) * n_lon + (i + 1) % n_lon
            faces.append([a, b, d])
            faces.append([a, d, c])
    return Mesh(verts, np.array(faces))


def make_cylinder_mesh(
    n_theta: int = 20, n_z: int = 10, radius: float = 1.0, height: float = 2.0
) -> Mesh:
    """Siatka walca (bez den). Oczekiwane: jedna krzywizna główna ≈0
    (wzdłuż osi z), druga ≈1/radius (obwodowa) — Test 3. Wiersze przy
    z=0 i z=height mają niepełny 1-ring (brak sąsiada z jednej strony)
    — pomiń je w testach."""
    thetas = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(0.0, height, n_z)
    verts = []
    for z in zs:
        for th in thetas:
            verts.append([radius * np.cos(th), radius * np.sin(th), z])
    verts = np.array(verts)
    faces = []
    for j in range(n_z - 1):
        for i in range(n_theta):
            a = j * n_theta + i
            b = j * n_theta + (i + 1) % n_theta
            c = (j + 1) * n_theta + i
            d = (j + 1) * n_theta + (i + 1) % n_theta
            faces.append([a, b, d])
            faces.append([a, d, c])
    return Mesh(verts, np.array(faces))
