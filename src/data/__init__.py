"""Datasets package. Importing it side-effect-registers every concrete dataset."""

from data import base  # noqa: F401  -- registers _TrivialDataset
from data import tree_cross_path  # noqa: F401  -- registers TreeCrossPath{Small,Medium,Large}
from data import cartesian_products  # noqa: F401  -- registers Cartesian-product datasets
