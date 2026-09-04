# TIMDR-Geometry-Formalism

Numeryczna implementacja Aksjomatów G8-G9 gałęzi geometrycznej TIMDR:
dyskretny operator kształtu (Weingarten) na trójkątnej siatce 3D,
domykający numerycznie związek skrętu powierzchniowego z krzywizną,
nazwany w [`Axioms_G_TIMDR_Geometry.md`](../GIA-TIMDR/docs/theory/Axioms_G_TIMDR_Geometry.md)
(Aksjomat G4) i domknięty tam analitycznie (Aksjomaty G8-G9).

Analogiczny cel jak `TIMDR-Math-Formalism` dla gałęzi sygnałowej: teoria
żyje w GIA-TIMDR, tu żyje działający, testowalny kod, który ją
implementuje — nie osobna, konkurencyjna definicja.

![Potok algorytmu: siatka 3D → normalne wierzchołkowe → rzut na styczną → dopasowanie MNK → wartości własne, rozgałęziające się na T_S empiryczny i T_S przewidywany, porównywane na końcu](docs/diagram_pipeline.svg)

![Trzy powierzchnie testowe: płaszczyzna z równoległymi normalnymi (κ≈0), sfera z normalnymi rozchodzącymi się promieniście (κ≈1/R), walec z krzywizną zero wzdłuż osi i 1/r obwodowo](docs/diagram_surfaces.svg)

## Co to liczy

Dla wierzchołka `p` trójkątnej siatki `S⊂ℝ³`:

1. Normalną wierzchołkową `n(p)` (średnia normalnych sąsiednich
   trójkątów, ważona polem).
2. Dyskretny operator kształtu `S_p: ℝ³→ℝ³` (a właściwie
   `T_pS→T_pS`) metodą najmniejszych kwadratów na 1-ringu sąsiadów —
   Aksjomat G9b.
3. Krzywizny główne `κ1, κ2` jako wartości własne `S_p` (po
   symetryzacji), krzywiznę średnią `H` i Gaussa `K`.
4. Skręt powierzchniowy `T_S(p)` dwoma sposobami: bezpośrednio z
   Aksjomatu G3 (`‖n(p+Δp)−n(p)‖`, funkcja `T_S_empirical`) i
   przewidziany z operatora kształtu wg Aksjomatu G9c
   (`‖Δp‖·‖S_p(Δ̂p)‖`, funkcja `T_S_predicted`) — do porównania.

**Konwencja znaku:** `S_p := -D_v n(p)` (Aksjomat G9a). Dla wypukłej
powierzchni z normalną na zewnątrz (np. sfera) to daje **ujemne**
wartości własne. Jedyne użycie tego operatora gdzie indziej w
formalizmie (Aksjomat G8d, ograniczenie Lipschitza) zależy tylko od
`κ_max = max(|κ1|,|κ2|)` — moduł i jego testy sprawdzają magnitudy, nie
znak.

## 🔧 Instalacja

```
pip install -r requirements.txt
```

Wymaga tylko `numpy` (Python 3.9+).

## 🚀 Szybki start

```python
from timdr_geometry import make_sphere_mesh, vertex_normals, one_ring, discrete_shape_operator, kappa_max

mesh = make_sphere_mesh(n_lat=24, n_lon=24, radius=2.0)
normals = vertex_normals(mesh)
rings = one_ring(mesh)

op = discrete_shape_operator(mesh, normals, point_idx=100, rings=rings)
print(op.principal_curvatures)   # oczekiwane: ~[-0.5, -0.5] (1/radius, ze znakiem -Dn)
print(kappa_max(op))             # ~0.5 — stała Lipschitza z Aksjomatu G8d
```

## 🧪 Testy

```
pip install pytest
pytest tests/ -v
```

Cztery testy stabilności — dokładnie te, których brakowało wg Aksjomatu
G7c — na przypadkach znanych analitycznie:

| Test | Siatka | Oczekiwany wynik |
|---|---|---|
| `TestPlane` | płaszczyzna | `S_p≡0`, `T_S≡0` wszędzie |
| `TestSphere` | sfera promienia R | obie krzywizny główne `≈1/R`, `K=κ1κ2≈1/R²>0` |
| `TestCylinder` | walec promienia r | jedna krzywizna `≈0` (oś), druga `≈1/r` (obwód) |
| `TestMeshRefinement` | sfera, rosnąca rozdzielczość | błąd krzywizny i błąd `T_S_predicted` vs `T_S_empirical` maleją przy zagęszczaniu siatki |

**⚠️ Nie uruchomione w tej sesji.** Sandbox bash był niedostępny
(RPC pipe closed) przez cały czas pisania tego modułu — matematyka
została prześledzona ręcznie krok po kroku, a tolerancje w testach są
celowo szerokie, ale to nie zastępuje faktycznego `pytest tests/ -v`.
Uruchom testy przed zaufaniem tym liczbom.

## ⚠️ Czego to NIE robi (status wg Aksjomatu G7)

Ten moduł domyka **numerycznie** to, co Aksjomaty G8-G9 domknęły
**analitycznie** — implementację operatora kształtu na konkretnej
siatce. To NIE zamyka całego Aksjomatu G7c:

- **(1) pełna definicja `W_S`** — częściowo domknięte (różniczkowa +
  dyskretna wersja tutaj).
- **(2) formalna przestrzeń powierzchni** — wciąż otwarte, ten moduł
  operuje na konkretnych siatkach, nie na przestrzeni wszystkich
  dopuszczalnych powierzchni.
- **(3) testy empiryczne** — testy tego repo są na siatkach
  SYNTETYCZNYCH o znanej analitycznie odpowiedzi (płaszczyzna/sfera/
  walec), nie na rzeczywistych danych geometrycznych (skan 3D, dane
  pomiarowe) — w odróżnieniu od gałęzi M/S, gdzie
  `TIMDR-Math-Formalism/docs/REAL_DATA_VALIDATION.md` dokumentuje test
  na realnych danych. Odpowiednika tego dla gałęzi G nie ma.
- **(4) niezależna walidacja** — wciąż otwarte.

Gałąź geometryczna pozostaje **koncepcyjna** (Aksjomat G7a) mimo tego
modułu — ten kod jest jednym z czterech wymaganych kroków, nie
wszystkimi czterema.

## 📄 Licencja

MIT — patrz [LICENSE](LICENSE).
