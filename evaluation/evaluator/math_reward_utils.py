"""
evaluation.math_reward_utils
----------------------------
Utility helpers to compute *math correctness rewards* for RLHF/RL
fine-tuning runs (e.g. GRPO, PPO).  The logic is adapted from the Minerva
paper and the `lm-evaluation-harness` MATH task, wrapped so you can call:

>>> from evaluation.math_reward_utils import compute_score
>>> reward_dict = compute_score(solution_str, ground_truth)
>>> reward = reward_dict["score"]   # +1.0 if correct, -1.0 otherwise

Only the Python std-lib (``re``, ``typing``) is required, so this file is
import-safe even inside restricted sandboxes.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Low-level helpers ----------------------------------------------------------
# ---------------------------------------------------------------------------


def last_boxed_only_string(string: str) -> Optional[str]:
    """Return the *last* ``\\boxed{...}`` substring from *string* (inclusive).

    If no such substring exists, returns ``None``.
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx: Optional[int] = None
    num_left_braces_open = 0

    # Walk forward to find the matching right brace.
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Strip the leading ``\\boxed{`` and trailing ``}`` from *s*.

    >>> remove_boxed("\\boxed{42}")
    '42'
    """
    left = "\\boxed{"
    assert s.startswith(left), f"box error: {s}"
    assert s.endswith("}"), f"box error: {s}"
    return s[len(left) : -1]


# ---------------------------------------------------------------------------
# Normalisation rules -------------------------------------------------------
# ---------------------------------------------------------------------------

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\\n}",
    "\\text{}",
    r"\\mathrm{th}",
    r"^\\circ",
    r"^{\\circ}",
    r"\\;",
    r",\\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalise *final_answer* so that superficial formatting differences
    do not affect equality comparison.
    """
    # Keep only the rhs if a '=' exists.
    final_answer = final_answer.split("=")[-1]

    # Simple textual replacements.
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Preserve *one* math expression if present, drop surplus.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"$\\3$", final_answer)

    # Remove \\text{...} etc.
    final_answer = re.sub(r"(\\text\\{)(.*?)(\\})", r"\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\\{)(.*?)(\\})", r"\\2", final_answer)
    final_answer = re.sub(r"(\\overline\\{)(.*?)(\\})", r"\\2", final_answer)

    # Drop boxed wrapper entirely.
    final_answer = re.sub(r"(\\boxed\\{)(.*)(\\})", r"\\2", final_answer)

    # Normalise shorthand TeX like \\fracab → \\frac{a}{b}
    final_answer = re.sub(r"(frac)([^\\{])(.)", r"frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^\\{])", r"sqrt{\\2}", final_answer)

    # Remove $ delimiters introduced above.
    final_answer = final_answer.replace("$", "")

    # Remove commas in numbers (e.g. 1,000 → 1000) if the rest is digits.
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


# ---------------------------------------------------------------------------
# Verification helpers ------------------------------------------------------
# ---------------------------------------------------------------------------


def is_correct_minerva(
    solution_str: str,
    gt: str,
    *,
    gt_need_extract: bool = False,
    answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)",
) -> Tuple[bool, str]:
    """Return *(correct?, normalised_prediction)* under non-strict rules.

    * Extracts the *last* line that matches *answer_pattern* inside
      *solution_str*.
    * Normalises both prediction and ground-truth (optionally unboxing *gt*).
    """
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Optionally unbox & normalise the gt.
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(
    pred: str,
    gt: str,
    *,
    pause_tokens_index: Optional[list[int]] = None,
) -> Tuple[int, Optional[str]]:
    """Strict criteria: prediction **must** contain a ``\\boxed{...}`` equal to *gt*.

    Returns (*score*, *extracted_pred*) where *score* is +1 if correct else -1.
    """
    # Focus only on last ~100 tokens; helps speed and reduces false positives.
    if pause_tokens_index is not None and len(pause_tokens_index) == 4:
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return (1 if extracted_pred == gt else -1), extracted_pred


def verify(
    solution_str: str,
    answer: str,
    *,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> Tuple[bool, Optional[str]]:
    """Wrapper around the two criteria above."""
    if strict_box_verify:
        score, pred = is_correct_strict_box(
            solution_str,
            answer,
            pause_tokens_index=pause_tokens_index,
        )
        return score == 1, pred

    return is_correct_minerva(solution_str, answer)


# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------


def compute_score(
    solution_str: str,
    ground_truth: str,
    *,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> Dict[str, Any]:
    """Return a *dict* with keys

    * ``score`` : ``+1.0`` if prediction correct else ``-1.0``
    * ``acc``   : boolean same as *(score > 0)*
    * ``pred``  : the normalised predicted answer (for logging)
    """
    # Clip very long generations; longest answer in MATH-500 is 159 chars.
    solution_str = solution_str[-300:]

    correct, pred = verify(
        solution_str,
        ground_truth,
        strict_box_verify=strict_box_verify,
        pause_tokens_index=pause_tokens_index,
    )

    reward = 1.0 if correct else -1.0

    return {
        "score": reward,
        "acc": correct,
        "pred": pred,
    }


__all__ = [
    "compute_score",
    "verify",
    "is_correct_minerva",
    "is_correct_strict_box",
    "normalize_final_answer",
    "last_boxed_only_string",
    "remove_boxed",
]
