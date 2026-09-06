"""Bounded observation-preserving PDE feedback controller."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
import math


@dataclass
class FeedbackState:
    round_id: int
    score: float
    accepted: bool
    reason: str


class BoundedFeedbackController:
    """Accept only measured improvements; original observations are immutable."""

    def __init__(self, observation_fingerprint: str, *, max_rounds=3, min_improvement=1e-3):
        if not observation_fingerprint or max_rounds < 1 or min_improvement < 0:
            raise ValueError("invalid feedback controller configuration")
        self.observation_fingerprint = str(observation_fingerprint)
        self.max_rounds = int(max_rounds)
        self.min_improvement = float(min_improvement)
        self._best_score = float("inf")
        self._best_state = None
        self.history: list[FeedbackState] = []

    def propose(self, round_id, score, state, observation_fingerprint):
        if round_id >= self.max_rounds:
            return False, "budget_exhausted"
        if str(observation_fingerprint) != self.observation_fingerprint:
            raise ValueError("feedback cannot modify the original observation contract")
        if not math.isfinite(float(score)) or score < 0:
            raise ValueError("score must be finite and non-negative")
        improved = self._best_state is None or score <= self._best_score - self.min_improvement
        accepted = bool(improved)
        reason = "improved" if accepted else "below_improvement_threshold"
        self.history.append(FeedbackState(int(round_id), float(score), accepted, reason))
        if accepted:
            self._best_score = float(score)
            self._best_state = copy.deepcopy(state)
        return accepted, reason

    def result(self):
        return {"observation_fingerprint": self.observation_fingerprint,
                "best_score": None if self._best_state is None else self._best_score,
                "best_state": copy.deepcopy(self._best_state),
                "history": [asdict(item) for item in self.history],
                "stopped": len(self.history) >= self.max_rounds}

    def run(self, initial_state, solve, integrate, render, score, observation_fingerprint=None):
        """Run bounded PDE -> integration -> render rounds with immutable observations."""
        fingerprint = self.observation_fingerprint if observation_fingerprint is None else observation_fingerprint
        state = copy.deepcopy(initial_state)
        for round_id in range(self.max_rounds):
            solved = solve(copy.deepcopy(state), round_id)
            integrated = integrate(copy.deepcopy(solved), round_id)
            rendered = render(copy.deepcopy(integrated), round_id)
            metric = float(score(rendered, round_id))
            accepted, _ = self.propose(round_id, metric, integrated, fingerprint)
            if accepted:
                state = copy.deepcopy(integrated)
        return self.result()
