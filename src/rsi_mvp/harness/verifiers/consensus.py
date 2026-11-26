"""Label-free ConsensusVerifier: K-sample majority voting with symbolic clustering & entropy."""

from __future__ import annotations

import math

from rsi_mvp.harness.adapters.math import extract_boxed_answer
from rsi_mvp.harness.verifiers.base import register_verifier
from rsi_mvp.harness.verifiers.ground_truth import math_answers_equivalent
from rsi_mvp.schemas import TaskSolution, TaskSpec, VerifyResult


def _answers_match(domain: str, a: str, b: str) -> bool:
    if domain == "math":
        return math_answers_equivalent(a, b)
    return " ".join(a.strip().lower().split()) == " ".join(b.strip().lower().split())


def compute_shannon_entropy(counts: list[int]) -> float:
    """Compute Shannon entropy (in bits) over equivalence cluster counts."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return round(entropy, 4)


class ConsensusVerifier:
    """Label-free self-consistency verifier (majority voting + confidence gating).

    Clusters ``K`` sampled answers into equivalence classes (using symbolic SymPy
    equivalence for ``math`` tasks), emits the majority answer as ``pseudo_label``,
    and rejects low-consensus tasks to prevent echo-chamber noise from entering
    reflection or LoRA fine-tuning.
    """

    mode_name = "consensus"
    is_label_free = True

    def __init__(self, min_consensus: float = 0.6) -> None:
        self.min_consensus = min_consensus

    def verify(self, spec: TaskSpec, solution: TaskSolution) -> VerifyResult:
        if solution.error:
            return VerifyResult(
                task_id=spec.task_id,
                passed=False,
                score=0.0,
                confidence=0.0,
                pseudo_label=None,
                is_label_free=True,
                failure_type="agent_crash",
                diagnostics=solution.error[:1024],
                tokens_used=solution.total_tokens,
                elapsed_sec=solution.elapsed_sec,
            )

        extracted_list: list[str] = []
        for out in solution.outputs:
            ans = out.extracted_answer.strip()
            if not ans and spec.domain == "math":
                ans = extract_boxed_answer(out.raw_text).strip()
            elif not ans:
                ans = out.raw_text.strip()
            if ans:
                extracted_list.append(ans)

        k_total = len(solution.outputs)
        if not extracted_list or k_total == 0:
            return VerifyResult(
                task_id=spec.task_id,
                passed=False,
                score=0.0,
                confidence=0.0,
                pseudo_label=None,
                is_label_free=True,
                failure_type="low_consensus",
                diagnostics="No non-empty candidate answers extracted.",
                tokens_used=solution.total_tokens,
                elapsed_sec=solution.elapsed_sec,
            )

        # Group into equivalence clusters: list of (representative, count)
        clusters: list[tuple[str, int]] = []
        for ans in extracted_list:
            matched_idx: int | None = None
            for idx, (rep, _cnt) in enumerate(clusters):
                if _answers_match(spec.domain, ans, rep):
                    matched_idx = idx
                    break
            if matched_idx is not None:
                rep, cnt = clusters[matched_idx]
                clusters[matched_idx] = (rep, cnt + 1)
            else:
                clusters.append((ans, 1))

        clusters.sort(key=lambda pair: (-pair[1], len(pair[0])))
        best_answer, best_votes = clusters[0]
        consensus_ratio = round(best_votes / k_total, 4)
        entropy = compute_shannon_entropy([c for _, c in clusters])
        passed = consensus_ratio >= (self.min_consensus - 1e-9)

        diag = (
            f"clusters={len(clusters)} top_votes={best_votes}/{k_total} "
            f"consensus={consensus_ratio:.2f} entropy={entropy:.3f}b"
        )
        return VerifyResult(
            task_id=spec.task_id,
            passed=passed,
            score=consensus_ratio if passed else 0.0,
            confidence=consensus_ratio,
            pseudo_label=best_answer if passed else None,
            is_label_free=True,
            failure_type="none" if passed else "low_consensus",
            diagnostics=diag,
            tokens_used=solution.total_tokens,
            elapsed_sec=solution.elapsed_sec,
        )


register_verifier("consensus", ConsensusVerifier)
