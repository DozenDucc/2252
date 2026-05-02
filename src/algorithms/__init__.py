"""Algorithms package. Importing it side-effect-registers every concrete algorithm."""

from algorithms import base  # noqa: F401  -- registers _AllZerosAlgorithm
from algorithms import standard  # noqa: F401  -- registers StandardSpectralClustering
from algorithms import cheeger_sweep_cut  # noqa: F401  -- registers CheegerSweepCut
from algorithms import higher_order_cheeger  # noqa: F401  -- registers HigherOrderCheeger
from algorithms import sweep_on_kth  # noqa: F401  -- registers SweepOnKth
