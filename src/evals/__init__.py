"""Evals package. Importing it side-effect-registers every concrete eval."""

from evals import base  # noqa: F401  -- registers _LabelAccuracyEval
from evals import conductance  # noqa: F401  -- registers ConductanceEval
from evals import ari  # noqa: F401  -- registers AdjustedRandIndexEval
from evals import ami  # noqa: F401  -- registers AdjustedMutualInfoEval
