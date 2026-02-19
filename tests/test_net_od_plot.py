import pytest
import pandas as pd
import sys
sys.path.insert(0, 'C:/GitHub/FCO')
from net_od_plot import map_regions_to_doses


def test_map_regions_to_doses_basic():
    """10 films, 5 doses -> 2 replicates each, reversed order."""
    doses = [0, 10, 50, 200, 500]
    n_regions = 10
    result = map_regions_to_doses(n_regions, doses)
    assert result[1] == 500
    assert result[2] == 500
    assert result[3] == 200
    assert result[9] == 0
    assert result[10] == 0


def test_map_regions_to_doses_variable():
    """6 films, 3 doses -> 2 replicates each."""
    doses = [0, 50, 500]
    n_regions = 6
    result = map_regions_to_doses(n_regions, doses)
    assert result[1] == 500
    assert result[2] == 500
    assert result[3] == 50
    assert result[4] == 50
    assert result[5] == 0
    assert result[6] == 0


def test_map_regions_to_doses_three_replicates():
    """9 films, 3 doses -> 3 replicates each."""
    doses = [0, 50, 500]
    n_regions = 9
    result = map_regions_to_doses(n_regions, doses)
    assert result[1] == 500
    assert result[3] == 500
    assert result[4] == 50
    assert result[6] == 50
    assert result[7] == 0
    assert result[9] == 0
