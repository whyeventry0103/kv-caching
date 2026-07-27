"""TV-matched full-cache noise control: the on-policy top rung of the causal ladder.

CONTRACT (mandatory, not optional). During full-cache free-running generation,
inject calibrated per-step perturbations ``(1 - eps) * p + eps * q`` whose induced
per-step TV is matched to the profile measured for method ``M`` under replay. The
test:

* if matched *generic* noise reproduces ``M``'s accuracy change, then the
  magnitude of readout distortion suffices and any compounding is generic;
* if it does not, ``M``'s damage is *structured* (e.g. anchor deletion, not
  noise) — the mechanism claim, proven interventionally, on-policy, with the
  generation feedback loop intact.

This simultaneously answers the off-policy objection to readout/TDP and gives the
ladder its strongest rung; it is treated as required, not a nice-to-have. Like
every instrument here, its accuracy effect is reported with signed flip-mass so a
reproduced *improvement* is not mislabeled as damage.

This module currently exposes only the calibration objective (matching achieved
TV to the target profile). The injection/generation engine is a GPU/M7 deliverable.
"""

from __future__ import annotations

from collections.abc import Sequence


def calibration_error(target: Sequence[float], achieved: Sequence[float]) -> float:
    """Mean absolute per-step gap between target and achieved TV profiles."""
    if len(target) != len(achieved):
        raise ValueError("TV profiles must have equal length")
    return (
        sum(abs(a - b) for a, b in zip(target, achieved, strict=True)) / len(target)
        if target
        else 0.0
    )


def window_means(
    profile: Sequence[float], *, first_window: int = 50
) -> dict[str, float]:
    """Pre-eviction / first-50 / later means of an onset-aligned TV profile.

    Same three windows the paper reports for the measured profiles, so a
    calibration check and a published number cannot be computed differently.
    ``profile`` is onset-aligned: index 0 is the onset step, so there is no
    pre-eviction segment inside it and that window is reported as 0.0 for
    interface symmetry with the measured profiles.
    """
    if not profile:
        raise ValueError("empty profile")
    head = list(profile[:first_window])
    tail = list(profile[first_window:])
    return {
        "before": 0.0,
        "first50": sum(head) / len(head),
        "later": (sum(tail) / len(tail)) if tail else 0.0,
    }


def match_gate(
    target: Sequence[float],
    achieved: Sequence[float],
    *,
    window_tolerance: float = 0.10,
    shape_tolerance: float = 0.25,
    first_window: int = 50,
) -> dict[str, object]:
    """Decide whether injected noise reproduces a target TV profile.

    **This gate exists to be run BEFORE any accuracy is read off the noise
    control, and the accuracy must not be read if it fails.** The control's
    entire logic is "matched noise, therefore any accuracy difference is
    attributable to structure rather than magnitude". If the match is not
    demonstrated, a null result means the calibration failed and a positive
    result means nothing at all -- in both cases the experiment has not been
    run, and reading the number anyway converts a broken instrument into a
    finding. This repo has one precedent for that failure mode already: the
    R-KV parity harness reported a perfect match while the live press ran the
    wrong window, and only a gate at the real boundary caught it.

    Two conditions, both required, because either alone is satisfiable by an
    obviously wrong profile:

    * **window means** within ``window_tolerance`` (relative) on the first-50
      and later windows -- magnitude;
    * **growth shape**, the later/first-50 ratio, within ``shape_tolerance``
      (relative) -- a flat profile with the right average is not a match to a
      compounding one, and compounding is the property under test.
    """
    if len(target) != len(achieved):
        raise ValueError("TV profiles must have equal length")
    tw = window_means(target, first_window=first_window)
    aw = window_means(achieved, first_window=first_window)

    def _rel(a: float, b: float) -> float:
        return abs(a - b) / abs(b) if b else (0.0 if a == b else float("inf"))

    window_errors = {name: _rel(aw[name], tw[name]) for name in ("first50", "later")}
    windows_ok = all(e <= window_tolerance for e in window_errors.values())

    def _growth(w: dict[str, float]) -> float:
        return w["later"] / w["first50"] if w["first50"] else float("inf")

    target_growth, achieved_growth = _growth(tw), _growth(aw)
    shape_error = _rel(achieved_growth, target_growth)
    shape_ok = shape_error <= shape_tolerance

    reasons = []
    if not windows_ok:
        reasons.append(
            "window means outside tolerance: "
            + ", ".join(f"{k} rel err {v:.3f}" for k, v in window_errors.items())
        )
    if not shape_ok:
        reasons.append(
            f"growth shape outside tolerance: target {target_growth:.2f}x, "
            f"achieved {achieved_growth:.2f}x, rel err {shape_error:.3f}"
        )
    return {
        "passed": bool(windows_ok and shape_ok),
        "target_windows": tw,
        "achieved_windows": aw,
        "window_relative_errors": window_errors,
        "target_growth": target_growth,
        "achieved_growth": achieved_growth,
        "shape_relative_error": shape_error,
        "mean_abs_error": calibration_error(target, achieved),
        "reasons": reasons,
        "contract": (
            "Accuracy from this control MUST NOT be reported unless passed is "
            "true. A failed match means the experiment did not run."
        ),
    }
