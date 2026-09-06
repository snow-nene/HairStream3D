import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.scenario_matrix import scenario_matrix, scenario_record, validate_matrix


def test_scenario_matrix_keeps_fixed_inputs_across_modes():
    scenarios = scenario_matrix("roots", seed=42, budget=10, configurations={"baseline": {}, "feedback": {"rounds": 2}})
    records = [scenario_record(item, {"loss": i}) for i, item in enumerate(scenarios)]
    report = validate_matrix(records)
    assert report == {"fixed_roots": True, "fixed_seed": True, "fixed_budget": True, "all_recorded": True}


def test_empty_matrix_is_rejected():
    with pytest.raises(ValueError):
        scenario_matrix("roots", seed=1, budget=1, configurations={})
