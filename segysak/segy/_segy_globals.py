
from collections import defaultdict

_SEGY_MEASUREMENT_SYSTEM = defaultdict(lambda: 0)
_SEGY_MEASUREMENT_SYSTEM[1] = "m"
_SEGY_MEASUREMENT_SYSTEM[2] = "ft"
_ISEGY_MEASUREMENT_SYSTEM = defaultdict(lambda: 0, m=1, ft=2)
