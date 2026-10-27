"""Decision engine: risk score (0-100) -> decision / confidence / reason.

Thresholds come from the lifted risk config (``model/config.json`` via
``app.config.RISK_CONFIG``) and are used verbatim — this module moves the
threshold logic out of ``main.py`` without changing behaviour:

  score < approve (40)  -> Approve, Low confidence
  score < review  (70)  -> Review,  Medium confidence
  score >= review (70)  -> Reject,  High confidence (< 90) / Very High (>= 90)

Note: the build plan's prose says 0.3/0.7 on a 0-1 scale, but the config
lifted from the original gateway repo says 40/70/90 on the 0-100 scale.
The config wins: behaviour is identical to Week 1. Do not "fix" the
thresholds to 30/70 without an explicit decision — that would silently
reclassify scores in [30, 40).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import RISK_CONFIG


@dataclass(frozen=True)
class Decision:
    decision: str
    confidence: str
    reason: str


def decide(risk_score: float, config: dict | None = None) -> Decision:
    """Decide from a 0-100 risk score using the lifted config thresholds."""
    cfg = config or RISK_CONFIG
    thresholds = cfg["risk_thresholds"]
    if risk_score < thresholds["approve"]:
        return Decision(
            cfg["decisions"]["approve"],
            cfg["confidence"]["low"],
            cfg["messages"]["approve"],
        )
    if risk_score < thresholds["review"]:
        return Decision(
            cfg["decisions"]["review"],
            cfg["confidence"]["medium"],
            cfg["messages"]["review"],
        )
    return Decision(
        cfg["decisions"]["reject"],
        cfg["confidence"]["high"]
        if risk_score < thresholds["high"]
        else cfg["confidence"]["very_high"],
        cfg["messages"]["reject"],
    )


def recommendation_for(decision: str) -> str:
    """Merchant recommendation matching a decision label.

    The plan's vocabulary is approve / review / block; the decision labels
    are Approve / Review / Reject. "Reject" maps to the "block" action.
    """
    return {"Approve": "approve", "Review": "review", "Reject": "block"}.get(
        decision, "review"
    )
