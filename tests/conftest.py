"""Shared pytest configuration.

Two safety nets are installed here, before any test imports ``blackholes``:

1. **Runtime type checking.** ``beartype_package`` instruments every function in
   the package, so a call with arguments that violate the annotations fails
   loudly during tests instead of silently computing garbage.
2. **Hypothesis profiles.** SymPy arithmetic is slow, so the default profile
   disables per-example deadlines; set ``HYPOTHESIS_PROFILE=thorough`` for a
   longer property-based search.
"""

import os

from beartype.claw import beartype_package
from hypothesis import settings

beartype_package("blackholes")

settings.register_profile("default", deadline=None, max_examples=40)
settings.register_profile("thorough", deadline=None, max_examples=500)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
