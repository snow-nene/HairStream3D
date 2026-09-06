import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.feedback_controller import BoundedFeedbackController


def test_feedback_accepts_improvement_and_rolls_back_worse_round():
    controller = BoundedFeedbackController("obs", max_rounds=3, min_improvement=.1)
    assert controller.propose(0, 1.0, {"field": 0}, "obs")[0]
    assert not controller.propose(1, .95, {"field": 1}, "obs")[0]
    assert controller.propose(2, .7, {"field": 2}, "obs")[0]
    result = controller.result()
    assert result["best_state"] == {"field": 2}
    assert len(result["history"]) == 3


def test_feedback_rejects_observation_mutation():
    controller = BoundedFeedbackController("obs")
    with pytest.raises(ValueError):
        controller.propose(0, 1.0, {}, "changed")


def test_feedback_runs_bounded_pipeline_and_keeps_best_state():
    controller = BoundedFeedbackController("obs", max_rounds=2, min_improvement=0.0)
    result = controller.run(0, lambda x, _: x + 1, lambda x, _: x * 2,
                            lambda x, _: {"value": x}, lambda out, _: abs(out["value"] - 4))
    assert result["best_state"] == 6
    assert len(result["history"]) == 2
