"""Put the repo root on sys.path.

The modules here are flat scripts rather than an installed package, so tests in
`tests/` need this to `import summ_net` and friends.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
