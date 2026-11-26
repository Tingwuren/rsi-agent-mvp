# 03. 详细设计与接口/数据契约说明书 (LLD)

- **文档版本**：v1.1
- **关联文档**：[02-architecture.md](./02-architecture.md) | [04-test-plan.md](./04-test-plan.md)

本文档定义面向**多任务、大数据集、双轨进化（Scaffold + LoRA）及有标签/无标签统一验证**的核心数据契约、抽象接口、算法与 CLI 规范。

---

## 1. 任务无关的核心数据模型（`src/rsi_mvp/schemas.py`）

### 1.1 通用任务规格（`TaskSpec` & `TaskInput`）

不再写死为 Python 代码字段，而是通过 `domain` 与 `payload` 适配任意任务，同时允许 `ground_truth = None` 支持无标签数据集：

```python
class TaskSpec(BaseModel):
    task_id: str
    domain: Literal["code", "math", "qa", "custom"]
    split: Literal["train", "dev", "holdout", "anchor"]
    category: str = "general"          # 分层抽样标签（如难度等级/子类别）
    prompt: str                        # 题目自然语言描述
    payload: dict[str, Any] = {}       # 任务域特有只读上下文（如 starter_code, entry_file）
    ground_truth: dict[str, Any] | None = None
    # 有标签示例：code -> {"test_code": "..."}；math -> {"answer": "42"}
    # 无标签数据：ground_truth 为 None
    timeout_sec: int = 20

class TaskInput(BaseModel):
    """透传给 Agent 沙箱的脱敏视图：永远不包含 ground_truth 与 split。"""
    task_id: str
    domain: str
    category: str
    prompt: str
    payload: dict[str, Any] = {}
    num_samples: int = 1               # 无标签共识或 Pass@K 场景下可要求 K 次采样
```

### 1.2 执行轨迹与多候选输出（`TaskSolution`）

```python
class TraceStep(BaseModel):
    step_index: int
    action_type: Literal["llm_call", "tool_call", "self_test", "final_answer", "error"]
    input_summary: str
    output_summary: str
    tool_name: str | None = None
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0

class CandidateOutput(BaseModel):
    raw_text: str                      # 完整回复（含可选思维链）
    extracted_answer: str              # 提取后的代码/最终答案
    auxiliary: dict[str, Any] = {}     # 如无标签代码模式下附带生成的自测代码 {"generated_tests": "..."}

class TaskSolution(BaseModel):
    task_id: str
    outputs: list[CandidateOutput]     # 单次采样长度为 1；共识/交叉验证模式长度为 K
    steps: list[TraceStep] = []
    error: str | None = None
    total_tokens: int = 0
    elapsed_sec: float = 0.0
```

### 1.3 统一验证与评测结果（`VerifyResult` & `SuiteEvalSummary`）

```python
class VerifyResult(BaseModel):
    task_id: str
    passed: bool                       # 二值判定（通过/有效）
    score: float                       # 连续分 [0.0, 1.0]
    confidence: float = 1.0            # 有标签恒为 1.0；无标签为共识比例/交叉一致度
    pseudo_label: str | None = None    # 无标签模式下选出的共识伪答案（供微调筛选）
    is_label_free: bool = False
    failure_type: Literal[
        "none", "wrong_answer", "assertion_error", "syntax_error",
        "runtime_error", "timeout", "low_consensus", "agent_crash"
    ] = "none"
    diagnostics: str = ""              # 截断后的报错堆栈或评审清单反馈（<= 4KB）
    tokens_used: int = 0
    elapsed_sec: float = 0.0

class SuiteEvalSummary(BaseModel):
    split: Literal["train", "dev_mini", "dev_full", "holdout", "anchor"]
    verifier_mode: str                 # 如 "ground_truth_code", "consensus", "code_test_co"
    mean_score: float
    pass_rate: float
    avg_confidence: float = 1.0
    output_entropy: float | None = None  # 多样性/防坍缩监控指标
    total_tasks: int
    passed_tasks: int
    total_tokens: int
    avg_elapsed_sec: float
    failure_breakdown: dict[str, int]
    results: list[VerifyResult]
```

### 1.4 元改进提案、LoRA 记录与双轨候选体（`CandidateRecord`）

```python
class FileEdit(BaseModel):
    target_path: str                   # 相对 agent/ 目录，如 "prompts.py"
    mode: Literal["replace_block", "overwrite_file", "create_file"]
    search: str | None = None
    replace: str

class MutationProposal(BaseModel):
    diagnosis: str
    hypothesis: str
    rationale: str
    edits: list[FileEdit]

class LoRAAdapterRecord(BaseModel):
    adapter_id: str                    # 如 "lora_0001"
    base_model: str                    # 如 "Qwen/Qwen3-8B"
    parent_adapter_id: str | None = None
    adapter_path: str                  # 位于 /mnt/data0/rsi-workspace/adapters/...
    train_samples: int
    method: Literal["sft_lora", "dpo_lora", "grpo_lora"] = "sft_lora"
    train_loss: float | None = None
    anchor_delta: float | None = None  # 相对微调前 Anchor 集分数变化
    accepted: bool = False
    created_at: str

class CandidateRecord(BaseModel):
    candidate_id: str                  # 如 "cand_0005"
    generation: int
    parent_id: str | None
    track: Literal["bootstrap", "scaffold", "weight_lora"]
    lora_adapter_id: str | None = None # 本候选体绑定的 LoRA 版本（None 表示纯基座）
    status: Literal[
        "accepted",
        "rejected_invalid",
        "rejected_smoke_failed",
        "rejected_mini_pruned",
        "rejected_anchor_drop",        # Anchor Gatekeeper 拦截
        "rejected_low_score",
        "error",
    ]
    snapshot_dir: str
    diff_path: str | None = None
    hypothesis: str = ""
    dev_mini_score: float | None = None
    dev_full_score: float | None = None
    anchor_score: float | None = None
    holdout_score: float | None = None
    output_entropy: float | None = None
    total_tokens: int = 0
    selection_count: int = 0
    failure_reason: str | None = None
    created_at: str
```

---

## 2. 核心可扩展接口契约（Protocols / ABCs）

### 2.1 数据供给层（`DatasetProvider`）

```python
class DatasetProvider(Protocol):
    def get_split(
        self,
        split: Literal["train", "dev", "holdout", "anchor"],
        max_samples: int | None = None,
        stratified: bool = True,
        seed: int = 42,
    ) -> list[TaskSpec]: ...

    def iter_batches(
        self, split: str, batch_size: int
    ) -> Iterator[list[TaskSpec]]:
        """面向十万级大数据集的流式/分片迭代接口（基于 HF datasets Arrow/Streaming）。"""
```

### 2.2 任务适配器与验证器（`TaskAdapter` & `Verifier`）

```python
class TaskAdapter(Protocol):
    domain: str
    def to_task_input(self, spec: TaskSpec, num_samples: int = 1) -> TaskInput: ...
    def extract_answer(self, raw_text: str, spec: TaskSpec) -> str: ...
    def format_for_sft(self, spec: TaskSpec, solution: TaskSolution, verify: VerifyResult) -> dict:
        """将高分/通过轨迹格式化为标准 messages SFT 训练样本。"""

class Verifier(Protocol):
    mode_name: str
    is_label_free: bool
    def verify(
        self, spec: TaskSpec, solution: TaskSolution, sandbox: SandboxExecutor
    ) -> VerifyResult: ...
```

内置 Verifier 行为规范：

1. **`GroundTruthCodeVerifier`**：提取代码至干净临时目录，注入 `spec.ground_truth["test_code"]` 跑隔离 `pytest`。
2. **`GroundTruthMathVerifier`**：使用 `math-verify` / `sympy` 对比提取答案与 `spec.ground_truth["answer"]`。
3. **`ConsensusVerifier`（无标签）**：对 `solution.outputs`（$K \ge 4$）按规范化等价类投票，计算 `consensus_ratio = max_votes / K`。若 `consensus_ratio >= min_consensus`（默认 0.6），标记 `passed=True` 并写入 `pseudo_label`；否则标 `low_consensus` 拒绝进入微调池。
4. **`CodeTestCoVerifier`（无标签代码）**：构建 $M$ 份候选代码 × $N$ 份候选测试的执行矩阵 $A \in \{0, 1\}^{M \times N}$，剔除全 0/全 1 的退化测试后，按行列一致度选取最优解并计算 `confidence`。

### 2.3 模型生命周期与微调后端（`ModelLifecycleManager` & `TrainerBackend`）

```python
class ModelLifecycleManager(Protocol):
    def ensure_serving(self, active_lora: LoRAAdapterRecord | None = None) -> BaseLLMClient:
        """确保推理服务就绪并挂载指定 LoRA（单卡模式按需拉起 vLLM）。"""

    def acquire_for_training(self) -> list[int]:
        """返回可用训练 GPU IDs；单卡分时模式下自动优雅停止 vLLM 并校验显存释放。"""

    def release_after_training(self, new_lora: LoRAAdapterRecord | None = None) -> None:
        """训练结束后清理显存，必要时重启 vLLM 并注册新 LoRA。"""

class TrainerBackend(Protocol):
    def train_lora(
        self,
        base_model: str,
        parent_adapter_path: str | None,
        train_jsonl: Path,
        output_dir: Path,
        gpu_ids: list[int],
        config: dict,
    ) -> LoRAAdapterRecord: ...
```

### 2.4 可变 Agent 标准契约（`agent/`）

```python
def solve(task: dict, llm_client: BaseLLMClient) -> dict:
    """输入 TaskInput dict，输出 TaskSolution dict（支持 num_samples >= 1）。"""

def propose_improvement(context: dict, llm_client: BaseLLMClient) -> dict:
    """输入父代源码、Dev 汇总、抽样失败 Trace 与近期历史，输出 MutationProposal dict。"""
```

---

## 3. 补丁校验、双轨选择与锚点门禁算法

### 3.1 补丁校验与双入口冒烟（`harness/patcher.py`）

1. 校验编辑文件数 `1 <= len(edits) <= max_files_per_mutation`（默认 3）。
2. 路径白名单 + 防 Symlink 逃逸校验（严格限定 `candidate_agent_dir` 内部）。
3. `replace_block` 要求 `search` 唯一精确匹配；应用后执行 `ast.parse()` 并生成 `patch.diff`。
4. 沙箱注入 `MockLLMClient` 分别验证 `solve()` 与 `propose_improvement()` 能在 5s 内返回合法 Schema。

### 3.2 开放式进化选择策略（`evolutionary_archive`）

对候选池中状态为 `accepted` 的个体 $i$ 计算父代采样权重（优先取 `dev_full_score`，若无则取 `dev_mini_score`）：

$$
w_i = \frac{\text{sigmoid}\!\left(\lambda \cdot (s_i - \bar{s})\right)}{1 + \alpha \cdot n_i}
$$

其中 $\lambda = 5.0$ 控制性能偏好，$\alpha = 0.5$ 惩罚重复抽样次数 $n_i$。同时支持切换为基线 `hill_climbing` 策略。

### 3.3 Anchor Gatekeeper 门禁规则

当启用无标签模式或完成一轮 LoRA 微调后，强制触发 `anchor` 集评估：

$$
\Delta_{\text{anchor}} = \text{Score}_{\text{anchor}}(\text{new}) - \text{Score}_{\text{anchor}}(\text{baseline})
$$

- 若 $\Delta_{\text{anchor}} < -\text{max\_anchor\_drop}$（默认 `-0.03`），直接标记 `rejected_anchor_drop`，阻止该 Scaffold 或 LoRA 成为活跃版本，防止静默退化与模型坍缩。

---

## 4. 实验配置文件规范（`configs/code_bigcodebench.yaml` 示例）

```yaml
experiment:
  name: "qwen3_8b_bigcodebench_dual_rsi"
  seed: 42
  workspace_root: "/mnt/data0/rsi-workspace"
  output_dir: "/mnt/data0/rsi-workspace/runs"

hardware:
  mode: "single_gpu_timeshare"          # single_gpu_timeshare | multi_gpu_split | external_api
  serving_gpus: [1]                     # 当前空闲 RTX 5090
  training_gpus: [1]                    # 多卡空闲时可改为 [2] 实现并行

model:
  base_model: "Qwen/Qwen3-8B"           # 或 /mnt/data0/rsi-workspace/models/Qwen3-8B
  vllm_host: "127.0.0.1"
  vllm_port: 8000
  max_model_len: 16384
  enable_lora: true
  temperature: 0.2

task:
  adapter: "code"                       # code | math | qa
  verifier: "ground_truth_code"         # ground_truth_code | ground_truth_math | consensus | code_test_co
  num_samples_per_task: 1               # 无标签共识模式下可设为 8

dataset:
  provider: "hf_local_parquet"          # jsonl | hf_local_parquet | hf_streaming
  path: "/mnt/data0/rsi-workspace/datasets/bigcodebench_240"
  mini_eval_size: 32                    # 大数据集快速筛选桶大小
  full_eval_top_k: 2                    # 仅 Top-K / 破纪录候选体触发全量 Dev 评测
  max_concurrency: 8

evolution:
  max_generations: 10
  selection_policy: "evolutionary_archive"
  holdout_eval_interval: 5
  early_stop_patience: 4
  max_files_per_mutation: 3
  max_failed_traces_in_context: 4

weight_adaptation:
  enabled: true                         # 设为 false 可退化为纯 Scaffold RSI 对照
  trigger_every_n_generations: 5
  method: "sft_lora"                    # sft_lora | dpo_lora | grpo_lora
  min_train_samples: 50
  min_confidence: 0.7                   # 无标签轨迹入库置信度下限
  lora_r: 16
  lora_alpha: 32
  learning_rate: 1.0e-4
  num_epochs: 2
  replay_ratio: 0.2                     # 混入基座原始数据防模型坍缩

guardrails:
  anchor_enabled: true
  max_anchor_drop: 0.03                 # Anchor 集最多允许抖动 3%
  max_total_tokens: 2000000

sandbox:
  backend: "subprocess"                 # subprocess | docker
  docker_image: "python:3.11-slim"
  task_timeout_sec: 30
  improve_timeout_sec: 90
  memory_limit: "1g"
```

---

## 5. CLI 命令规范（`src/rsi_mvp/cli.py`）

| 命令               | 说明                                                                                  | 示例                                                         |
| :----------------- | :------------------------------------------------------------------------------------ | :----------------------------------------------------------- |
| `rsi doctor`     | 检查 GPU 空闲状态、CUDA/`sm_120` 兼容性、`/mnt/data0` 磁盘余量与 `HF_HOME` 配置 | `rsi doctor`                                               |
| `rsi serve`      | 按配置在指定 GPU 启动/停止本地 vLLM（带 `--enable-lora`）                           | `rsi serve --config configs/code_bigcodebench.yaml`        |
| `rsi run`        | 启动或恢复双轨 RSI 进化实验                                                           | `rsi run --config configs/code_bigcodebench.yaml`          |
| `rsi eval`       | 对指定 `(snapshot, lora)` 组合运行 `dev/holdout/anchor` 评测                      | `rsi eval --snapshot ... --lora lora_0001 --split holdout` |
| `rsi train-lora` | 手动从指定实验 Trace 导出训练集并执行一轮单卡 LoRA 微调                               | `rsi train-lora --run runs/<exp_id>`                       |
| `rsi inspect`    | 展示双轨谱系表（Scaffold + LoRA 版本、Dev/Anchor/Holdout 曲线）                       | `rsi inspect --run runs/<exp_id>`                          |
| `rsi diff`       | 对比任意两个候选体的代码 Diff 与指标变化                                              | `rsi diff --run runs/<exp_id> cand_0000 cand_0005`         |
| `rsi report`     | 生成或刷新 `summary.md` 实验分析报告                                                | `rsi report --run runs/<exp_id>`                           |
