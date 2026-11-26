"""Labeled and label-free Verifier plugins."""

from rsi_mvp.harness.verifiers.base import (
    Verifier,
    get_verifier,
    list_verifiers,
    register_verifier,
)
from rsi_mvp.harness.verifiers.code_test_co import CodeTestCoVerifier
from rsi_mvp.harness.verifiers.consensus import ConsensusVerifier, compute_shannon_entropy
from rsi_mvp.harness.verifiers.ground_truth import (
    GroundTruthCodeVerifier,
    GroundTruthMathVerifier,
    math_answers_equivalent,
)

__all__ = [
    "CodeTestCoVerifier",
    "ConsensusVerifier",
    "GroundTruthCodeVerifier",
    "GroundTruthMathVerifier",
    "Verifier",
    "compute_shannon_entropy",
    "get_verifier",
    "list_verifiers",
    "math_answers_equivalent",
    "register_verifier",
]
