"""Core task-agnostic Pydantic v2 data schemas for RSI Agent MVP."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

DomainType = Literal["code", "math", "qa", "custom"]
SplitType = Literal["train", "dev", "holdout", "anchor"]
EvalSplitType = Literal["train", "dev_mini", "dev_full", "holdout", "anchor"]
FailureType = Literal[
    "none",
    "wrong_answer",
    "assertion_error",
    "syntax_error",
    "runtime_error",
    "timeout",
    "low_consensus",
    "agent_crash",
]
CandidateStatus = Literal[
    "accepted",
    "rejected_invalid",
    "rejected_smoke_failed",
    "rejected_mini_pruned",
    "rejected_anchor_drop",
    "rejected_low_score",
    "error",
]
TrackType = Literal["bootstrap", "scaffold", "weight_lora"]
TrainingMethod = Literal["sft_lora", "dpo_lora", "grpo_lora"]


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# 1. Task & Execution Schemas
# ---------------------------------------------------------------------------


class TaskInput(BaseModel):
    """Sanitized task view passed to the untrusted agent sandbox.

    Never contains ``ground_truth`` or ``split`` metadata.
    """

    task_id: str
    domain: DomainType
    category: str = "general"
    prompt: str
    payload: dict[str, Any] = Field(default_factory=dict)
    num_samples: int = Field(default=1, ge=1, le=64)


class TaskSpec(BaseModel):
    """Full task specification managed by the trusted harness.

    ``ground_truth`` may be ``None`` for unlabeled datasets.
    """

    task_id: str
    domain: DomainType
    split: SplitType
    category: str = "general"
    prompt: str
    payload: dict[str, Any] = Field(default_factory=dict)
    ground_truth: dict[str, Any] | None = None
    timeout_sec: int = Field(default=20, ge=1, le=600)

    def to_task_input(self, num_samples: int = 1) -> TaskInput:
        """Create a sanitized ``TaskInput`` without ground truth or split info."""
        return TaskInput(
            task_id=self.task_id,
            domain=self.domain,
            category=self.category,
            prompt=self.prompt,
            payload=dict(self.payload),
            num_samples=num_samples,
        )


class TraceStep(BaseModel):
    """Single step recorded inside an agent execution trajectory."""

    step_index: int = Field(ge=0)
    action_type: Literal["llm_call", "tool_call", "self_test", "final_answer", "error"]
    input_summary: str
    output_summary: str
    tool_name: str | None = None
    latency_ms: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)


class CandidateOutput(BaseModel):
    """One sampled output from an agent solve invocation."""

    raw_text: str
    extracted_answer: str
    auxiliary: dict[str, Any] = Field(default_factory=dict)


class TaskSolution(BaseModel):
    """Result returned by ``agent.solve``."""

    task_id: str
    outputs: list[CandidateOutput] = Field(min_length=1)
    steps: list[TraceStep] = Field(default_factory=list)
    error: str | None = None
    total_tokens: int = Field(default=0, ge=0)
    elapsed_sec: float = Field(default=0.0, ge=0.0)


# ---------------------------------------------------------------------------
# 2. Verification & Evaluation Schemas
# ---------------------------------------------------------------------------


class VerifyResult(BaseModel):
    """Per-task verification result for both labeled and label-free verifiers."""

    task_id: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    pseudo_label: str | None = None
    is_label_free: bool = False
    failure_type: FailureType = "none"
    diagnostics: str = ""
    tokens_used: int = Field(default=0, ge=0)
    elapsed_sec: float = Field(default=0.0, ge=0.0)


class SuiteEvalSummary(BaseModel):
    """Aggregated evaluation summary across a task split."""

    split: EvalSplitType
    verifier_mode: str
    mean_score: float = Field(ge=0.0, le=1.0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    avg_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    output_entropy: float | None = Field(default=None, ge=0.0)
    total_tasks: int = Field(ge=0)
    passed_tasks: int = Field(ge=0)
    total_tokens: int = Field(default=0, ge=0)
    avg_elapsed_sec: float = Field(default=0.0, ge=0.0)
    failure_breakdown: dict[str, int] = Field(default_factory=dict)
    results: list[VerifyResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_counts(self) -> SuiteEvalSummary:
        if self.passed_tasks > self.total_tasks:
            raise ValueError("passed_tasks cannot exceed total_tasks")
        return self


# ---------------------------------------------------------------------------
# 3. Self-Modification, LoRA & Archive Schemas
# ---------------------------------------------------------------------------


ALLOWED_EDIT_EXTENSIONS = frozenset({".py", ".txt", ".md", ".json"})


class FileEdit(BaseModel):
    """Single file edit proposed by the meta-agent against ``agent/``."""

    target_path: str
    mode: Literal["replace_block", "overwrite_file", "create_file"]
    search: str | None = None
    replace: str

    @field_validator("target_path")
    @classmethod
    def _validate_relative_safe_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/").strip()
        if not normalized:
            raise ValueError("target_path must not be empty")
        pure = PurePosixPath(normalized)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"target_path must be a safe relative path, got: {value!r}")
        if pure.suffix not in ALLOWED_EDIT_EXTENSIONS:
            raise ValueError(
                f"target_path extension {pure.suffix!r} not in allowed set "
                f"{sorted(ALLOWED_EDIT_EXTENSIONS)}"
            )
        return pure.as_posix()

    @model_validator(mode="after")
    def _validate_replace_block_fields(self) -> FileEdit:
        if self.mode == "replace_block" and not self.search:
            raise ValueError("search must be non-empty when mode == 'replace_block'")
        return self


class MutationProposal(BaseModel):
    """Structured improvement proposal returned by ``agent.propose_improvement``."""

    diagnosis: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    edits: list[FileEdit] = Field(min_length=1)


class LoRAAdapterRecord(BaseModel):
    """Metadata for a LoRA checkpoint produced by Track-B weight adaptation."""

    adapter_id: str
    base_model: str
    parent_adapter_id: str | None = None
    adapter_path: str
    train_samples: int = Field(ge=0)
    method: TrainingMethod = "sft_lora"
    train_loss: float | None = None
    anchor_delta: float | None = None
    accepted: bool = False
    created_at: str = Field(default_factory=utc_now_iso)


class CandidateRecord(BaseModel):
    """Persistent lineage entry combining scaffold snapshot and LoRA adapter."""

    candidate_id: str
    generation: int = Field(ge=0)
    parent_id: str | None = None
    track: TrackType
    lora_adapter_id: str | None = None
    status: CandidateStatus
    snapshot_dir: str
    diff_path: str | None = None
    hypothesis: str = ""
    dev_mini_score: float | None = Field(default=None, ge=0.0, le=1.0)
    dev_full_score: float | None = Field(default=None, ge=0.0, le=1.0)
    anchor_score: float | None = Field(default=None, ge=0.0, le=1.0)
    holdout_score: float | None = Field(default=None, ge=0.0, le=1.0)
    output_entropy: float | None = Field(default=None, ge=0.0)
    total_tokens: int = Field(default=0, ge=0)
    selection_count: int = Field(default=0, ge=0)
    failure_reason: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)


# ---------------------------------------------------------------------------
# 4. LLM & Experiment Configuration Schemas
# ---------------------------------------------------------------------------


class LLMResponse(BaseModel):
    """Normalized response returned by any ``BaseLLMClient`` implementation."""

    content: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    model: str = ""


class ExperimentMetaConfig(BaseModel):
    name: str = "rsi_mvp_exp"
    seed: int = 42
    workspace_root: str = "/mnt/data0/Projects/rsi-agent-mvp/.workspace"
    output_dir: str = "/mnt/data0/Projects/rsi-agent-mvp/.workspace/runs"


class HardwareConfig(BaseModel):
    mode: Literal["single_gpu_timeshare", "multi_gpu_split", "external_api"] = (
        "single_gpu_timeshare"
    )
    serving_gpus: list[int] = Field(default_factory=lambda: [1])
    training_gpus: list[int] = Field(default_factory=lambda: [1])


class ModelConfig(BaseModel):
    base_model: str = "Qwen/Qwen3-8B"
    vllm_host: str = "127.0.0.1"
    vllm_port: int = Field(default=8000, ge=1, le=65535)
    api_key_env: str = "RSI_LLM_API_KEY"
    base_url_env: str = "RSI_LLM_BASE_URL"
    max_model_len: int = Field(default=16384, ge=512)
    enable_lora: bool = True
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class TaskConfig(BaseModel):
    adapter: Literal["code", "math", "qa", "custom"] = "code"
    verifier: Literal[
        "ground_truth_code",
        "ground_truth_math",
        "consensus",
        "code_test_co",
        "rubric_critic",
    ] = "ground_truth_code"
    num_samples_per_task: int = Field(default=1, ge=1, le=64)


class DatasetConfig(BaseModel):
    provider: Literal["jsonl", "hf_local_parquet", "hf_streaming"] = "jsonl"
    path: str = "/mnt/data0/Projects/rsi-agent-mvp/.workspace/datasets/mbpp_plus"
    mini_eval_size: int = Field(default=32, ge=1)
    full_eval_top_k: int = Field(default=2, ge=1)
    max_concurrency: int = Field(default=8, ge=1, le=128)


class EvolutionConfig(BaseModel):
    max_generations: int = Field(default=10, ge=1)
    selection_policy: Literal["hill_climbing", "evolutionary_archive"] = "evolutionary_archive"
    holdout_eval_interval: int = Field(default=5, ge=1)
    early_stop_patience: int = Field(default=4, ge=1)
    max_files_per_mutation: int = Field(default=3, ge=1, le=20)
    max_failed_traces_in_context: int = Field(default=4, ge=1, le=20)


class WeightAdaptationConfig(BaseModel):
    enabled: bool = False
    trigger_every_n_generations: int = Field(default=5, ge=1)
    method: TrainingMethod = "sft_lora"
    min_train_samples: int = Field(default=50, ge=1)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    lora_r: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=1.0e-4, gt=0.0)
    num_epochs: int = Field(default=2, ge=1)
    replay_ratio: float = Field(default=0.2, ge=0.0, le=1.0)


class GuardrailsConfig(BaseModel):
    anchor_enabled: bool = True
    max_anchor_drop: float = Field(default=0.03, ge=0.0, le=1.0)
    max_total_tokens: int = Field(default=2_000_000, ge=1)
    min_free_disk_gb: float = Field(default=5.0, ge=0.5)


class SandboxConfig(BaseModel):
    backend: Literal["subprocess", "docker"] = "subprocess"
    docker_image: str = "python:3.11-slim"
    task_timeout_sec: int = Field(default=30, ge=1)
    improve_timeout_sec: int = Field(default=90, ge=1)
    memory_limit: str = "1g"


class ExperimentConfig(BaseModel):
    """Top-level validated experiment configuration."""

    experiment: ExperimentMetaConfig = Field(default_factory=ExperimentMetaConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    weight_adaptation: WeightAdaptationConfig = Field(default_factory=WeightAdaptationConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
