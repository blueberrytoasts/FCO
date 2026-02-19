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


def test_map_regions_to_doses_uneven_raises():
    """Raises ValueError when films don't divide evenly across dose levels."""
    with pytest.raises(ValueError, match="evenly divisible"):
        map_regions_to_doses(11, [0, 10, 50, 200, 500])


from net_od_plot import average_od_by_dose


def test_average_od_by_dose():
    """Average OD values per dose level across replicates."""
    df = pd.DataFrame({
        'region': [1, 2, 3, 4],
        'red_od':   [0.50, 0.52, 0.20, 0.22],
        'green_od': [0.40, 0.42, 0.10, 0.12],
        'blue_od':  [0.30, 0.32, 0.05, 0.07],
    })
    doses = [0, 500]
    result = average_od_by_dose(df, doses)
    # result is dict: dose -> {channel: mean_od}
    assert result[500]['red']   == pytest.approx(0.51)
    assert result[500]['green'] == pytest.approx(0.41)
    assert result[0]['red']     == pytest.approx(0.21)
    assert result[0]['blue']    == pytest.approx(0.06)


from net_od_plot import compute_net_od


def test_compute_net_od():
    """Net OD = post - pre for each dose and channel."""
    pre = {
        0:   {'red': 0.10, 'green': 0.15, 'blue': 0.20},
        500: {'red': 0.50, 'green': 0.55, 'blue': 0.60},
    }
    post = {
        0:   {'red': 0.12, 'green': 0.17, 'blue': 0.23},
        500: {'red': 0.80, 'green': 0.90, 'blue': 1.00},
    }
    result = compute_net_od(pre, post)
    assert result[0]['red']    == pytest.approx(0.02)
    assert result[0]['green']  == pytest.approx(0.02)
    assert result[500]['red']  == pytest.approx(0.30)
    assert result[500]['blue'] == pytest.approx(0.40)
