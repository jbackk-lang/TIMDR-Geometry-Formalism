import os
import sys

# Repo nie jest zainstalowanym pakietem — dorzuć katalog główny do
# sys.path, żeby `import timdr_geometry` działało niezależnie od tego,
# skąd odpalane jest pytest.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
