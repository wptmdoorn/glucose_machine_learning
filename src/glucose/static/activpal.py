"""src/glucose/static/activpal.py

This script contains all static variables used
for ActivPal processing and training.

"""

from typing import List

INCLUSION_VARIABLES: List[str] = [
    "time",
    "Sum(abs(dChannel1))",
    "Sum(abs(dChannel2))",
    "Sum(abs(dChannel3))"]
