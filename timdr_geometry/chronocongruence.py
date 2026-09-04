"""
timdr_geometry/chronocongruence.py

Kongruencja trajektorii Gamma(t,s) i operator ksztaltu dla chronoprocesu
-- numeryczna konstrukcja dla galezi G z Chronoprocesu
Xi=(T,x,Gamma,phi) opisanego w GIA-TIMDR/SKILL_timdr-signal-framework.md
("naprawa G": pojedyncza trajektoria jest krzywa 1D i nie ma operatora
ksztaltu; potrzeba RODZINY trajektorii {gamma_s}_{s in I},
Gamma(t,s)=gamma_s(t), zeby obraz S=Gamma(TxI) byl prawdziwa
powierzchnia 2D, na ktorej Axioms_G/G8/G9 dzialaja doslownie).

Ten modul NIE dodaje zadnej nowej matematyki do Axioms_G -- caly
aparat (vertex_normals, discrete_shape_operator, T_S_empirical,
T_S_predicted, kappa_max/mean_curvature/gaussian_curvature) z
weingarten.py jest uzyty BEZ ZMIAN. Jedyna nowosc to
`make_congruence_mesh`, ktora generalizuje wzorzec siatki siatkowej
(grid) juz uzyty w make_plane_mesh/make_sphere_mesh/make_cylinder_mesh
na DOWOLNA funkcje parametryzujaca gamma(t,s), plus trzy kanoniczne
przyklady kongruencji o znanej analitycznie krzywiznie (ta sama zasada
co plane/sphere/cylinder w weingarten.py).

CZEGO TO NIE ROBI -- granica zakresu, jawnie: ten modul mierzy
krzywizne ZEWNETRZNA (Weingarten/T_S) powierzchni zamiecionej przez
Gamma(t,s). NIE implementuje rozkladu ekspansja/scinanie/skret
(theta/sigma/omega) kongruencji geodezyjnych z rownania Raychaudhuriego
z OTW -- to byla wskazana ANALOGIA uzasadniajaca, ze T_S na takiej
powierzchni jest sensownym obiektem geometrycznym, NIE twierdzenie o
rownowaznosci T_S z ktoras z tamtych trzech wielkosci. Rownowaznosc (o
ile w ogole istnieje) wymagalaby osobnej, niewykonanej tu pracy.

Domena I (co dokladnie parametryzuje "sasiednie" trajektorie) byla
wczesniej flagowana jako pytanie MODELOWE, nie matematyczne. Trzy
przyklady ponizej sa KONKRETNYMI wyborami ilustrujacymi konstrukcje
(indeks polozenia na okregu / szerokosc geograficzna) -- nie twierdza,
ze to JEDYNY albo "poprawny" wybor I dla jakiegokolwiek realnego
zastosowania domenowego.

UWAGA O WYKONANIU: napisane w sesji bez dostepu do sandboxa bash --
oczekiwane krzywizny ponizej sa PRZEPISANE z juz ustalonych,
przetestowanych faktow analitycznych dla sfery/walca w weingarten.py
(nie nowa derywacja), ale ten konkretny plik + tests/test_chronocongruence.py
NIE zostaly jeszcze uruchomione. Uruchom
`pytest tests/test_chronocongruence.py -v` przed zaufaniem tym liczbom.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .weingarten import Mesh

__all__ = [
    "make_congruence_mesh",
    "flat_parallel_congruence",
    "cylindrical_congruence",
    "spherical_congruence",
]


GammaFn = Callable[[float, float], "np.ndarray"]


# ---------------------------------------------------------------------
# Budowa siatki z dowolnej parametryzacji Gamma(t,s)
# ---------------------------------------------------------------------

def make_congruence_mesh(
    gamma: GammaFn,
    t_values,
    s_values,
    t_periodic: bool = False,
    s_periodic: bool = False,
) -> Mesh:
    """Siatka trojkatna obrazu S=Gamma(TxI), Gamma(t,s)=gamma_s(t).

    t_values: probki T (np. "czas" wzdluz kazdej trajektorii).
    s_values: probki I (indeks, ktora sasiednia trajektoria).
    t_periodic/s_periodic: czy dana os siatki zawija sie (np. s=kat na
        okregu -> s_periodic=True; t=os czasu, zwykle otwarta ->
        t_periodic=False).

    Generalizuje dokladnie ten sam wzorzec indeksowania grid->trojkaty
    (a,b,c,d -> dwa trojkaty [a,b,d],[a,d,c]) co make_plane_mesh
    (t_periodic=s_periodic=False), make_cylinder_mesh
    (t_periodic=False, s_periodic=True, gamma=walec) i make_sphere_mesh
    (podobnie, gamma=sfera) -- te trzy funkcje w weingarten.py NIE
    zostaly przepisane na to wywolanie (zeby nie ryzykowac juz
    przetestowanego kodu), ale sa jego szczegolnymi przypadkami.
    """
    t_values = np.asarray(t_values, dtype=float)
    s_values = np.asarray(s_values, dtype=float)
    if t_values.size < 2 or s_values.size < 2:
        raise ValueError("t_values i s_values musza miec >= 2 elementy kazdy")

    verts = np.array(
        [np.asarray(gamma(t, s), dtype=float) for t in t_values for s in s_values]
    )
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("gamma(t,s) musi zwracac wektor 3D (x,y,z)")

    n_t = t_values.size
    n_s = s_values.size
    faces = []
    j_range = range(n_t) if t_periodic else range(n_t - 1)
    for j in j_range:
        j_next = (j + 1) % n_t
        i_range = range(n_s) if s_periodic else range(n_s - 1)
        for i in i_range:
            i_next = (i + 1) % n_s
            a = j * n_s + i
            b = j * n_s + i_next
            c = j_next * n_s + i
            d = j_next * n_s + i_next
            faces.append([a, b, d])
            faces.append([a, d, c])

    # Uwaga: `faces` jest gwarantowane niepuste tutaj -- t_values.size>=2
    # i s_values.size>=2 (sprawdzone wyzej) implikuja, ze j_range i
    # i_range maja kazdy >=1 element niezaleznie od t_periodic/s_periodic,
    # wiec petla wykonuje sie >=1 raz.
    return Mesh(verts, np.array(faces))


# ---------------------------------------------------------------------
# Trzy kanoniczne kongruencje o znanej analitycznie krzywiznie
# ---------------------------------------------------------------------

def flat_parallel_congruence(t: float, s: float) -> np.ndarray:
    """gamma_s(t) = (s, t, 0) -- rodzina rownoleglych, prostych
    trajektorii (kazda o stalym s, biegnaca wzdluz t), lezaca calkowicie
    w plaskiej plaszczyznie z=0. Zadna trajektoria nie zbliza/oddala sie
    od sasiadow (odleglosc miedzy gamma_s1(t) i gamma_s2(t) jest stala w
    t) I caly obraz jest plaski -- oczekiwana krzywizna zewnetrzna 0
    wszedzie (jak make_plane_mesh)."""
    return np.array([s, t, 0.0])


def cylindrical_congruence(t: float, s: float, radius: float = 1.0) -> np.ndarray:
    """gamma_s(t) = (r*cos(s), r*sin(s), t) -- rodzina trajektorii
    ulozonych po okregu (s = kat, sasiednia trajektoria = sasiedni kat),
    kazda biegnaca wzdluz wspolnej osi t przy stalym promieniu r.

    Wzdluz t (osiowo): trajektorie NIE zbiegaja sie ani nie rozbiegaja
    (rownolegle proste) -> oczekiwana krzywizna ~0.
    Wzdluz s (obwodowo): rodzina jest ulozona po okregu -> oczekiwana
    krzywizna ~1/r (co do wartosci bezwzglednej -- patrz konwencja znaku
    w weingarten.py). Identyczne fakty analityczne co make_cylinder_mesh
    w weingarten.py (tu: t<->z, s<->theta), zbudowane teraz jako
    Gamma(t,s) zamiast dedykowanego generatora."""
    return np.array([radius * np.cos(s), radius * np.sin(s), t])


def spherical_congruence(t: float, s: float, radius: float = 1.0) -> np.ndarray:
    """gamma_s(t) = (r*sin(t)*cos(s), r*sin(t)*sin(s), r*cos(t)) --
    t = szerokosc geograficzna (colatitude, 0<t<pi), s = dlugosc
    geograficzna (0<=s<2*pi). Rodzina trajektorii zbiega sie symetrycznie
    w obu kierunkach w poblizu biegunow (t->0 lub t->pi) i rozklada sie
    najszerzej na rowniku (t=pi/2) -- oczekiwana krzywizna ~1/r w OBU
    kierunkach glownych wszedzie (izotropowa zbieznosc/rozbieznosc,
    inaczej niz w cylindrical_congruence, gdzie jeden kierunek jest
    plaski). Identyczne fakty analityczne co make_sphere_mesh w
    weingarten.py, zbudowane tu jako Gamma(t,s)."""
    return np.array(
        [
            radius * np.sin(t) * np.cos(s),
            radius * np.sin(t) * np.sin(s),
            radius * np.cos(t),
        ]
    )
