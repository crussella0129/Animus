# Repor Findings — Animus

**Generated:** 2026-02-09
**Repos Analyzed:** 91 total (42 from prior sessions + 42 new + 7 inaccessible/redirected)

## Summary

Analysis of 91 external repositories reveals converging patterns for building productive local-first CLI coding agents. The most impactful findings center on: (1) grammar-constrained decoding to eliminate structural failures in small model output, (2) progressive disclosure memory to manage limited context windows, (3) variant/fallback systems to layer reliability around unreliable models, (4) skill-as-markdown with YAML frontmatter for extensibility, and (5) atomic task decomposition (2-5 minute units) as the key strategy for making small models productive. These findings map directly to Animus's existing architecture and suggest specific enhancements to its skills system, RAG pipeline, sub-agent orchestration, and model interaction layer.

---

## Findings by Goal

### Goal 1: Architectural Patterns

#### Pipeline/Step-Based Architecture
- **Finding:** The dominant pattern across coding agents is a step-based pipeline where each step is a pure function accepting the same dependency set and returning typed results.
- **Source:** AntonOsika/gpt-engineer (`gpt_engineer/core/default/steps.py`)
- **Relevance:** Animus's agent loop could adopt composable step functions (inject → plan → execute → validate) for testability.
- **Code Example:**
  ```python
  class BaseAgent(ABC):
      @abstractmethod
      def init(self, prompt: Prompt) -> FilesDict: ...
      @abstractmethod
      def improve(self, files_dict: FilesDict, prompt: Prompt) -> FilesDict: ...
  ```

#### Registry-Based Plugin Architecture
- **Finding:** VoltAgent maintains specialized registries (AgentRegistry, WorkflowRegistry, TriggerRegistry, MCPServerRegistry) with typed accessors and lifecycle management. Components self-register during initialization.
- **Source:** VoltAgent/voltagent (`packages/core/src/`)
- **Relevance:** Animus could use registries for skills, tools, and model backends instead of manual wiring.

#### Manager Delegation Pattern
- **Finding:** VoltAgent delegates to specialized managers (ToolManager, MemoryManager, SubAgentManager, LoggerProxy) — each encapsulates its domain completely. The Agent class coordinates but does not implement.
- **Source:** VoltAgent/voltagent (`packages/core/src/agent/`)
- **Relevance:** Animus's Agent class could delegate tool management, memory, and sub-agent coordination to dedicated manager classes.

#### Provider Registry with Decorator Discovery
- **Finding:** WrenAI uses `@provider("name")` decorator + `pkgutil.walk_packages` for zero-config provider registration.
- **Source:** Canner/WrenAI (`wren-ai-service/src/providers/loader.py`)
- **Relevance:** Animus could use this to auto-discover model backends, skills, and tool implementations.
- **Code Example:**
  ```python
  PROVIDERS = {}
  def provider(name: str):
      def wrapper(cls):
          PROVIDERS[name] = cls
          return cls
      return wrapper
  ```

#### Gateway/Variant Pattern
- **Finding:** TensorZero implements a gateway that normalizes 20+ LLM provider APIs, with variant-based experimentation (A/B testing prompts/models) and inference-time optimization (Best-of-N, DICL, Mixture-of-N).
- **Source:** tensorzero/tensorzero (`tensorzero-core/src/variant/`)
- **Relevance:** Animus could implement variant-based strategy selection: try local model → fallback to larger model → fallback to API.

#### Three-Layer Memory Hierarchy
- **Finding:** memU implements Resources (raw data) → MemoryItems (extracted facts with typed classification) → MemoryCategories (auto-organized topic groups) with embedding-based retrieval at every layer.
- **Source:** NevaMind-AI/memU (`src/memu/database/`)
- **Relevance:** Animus's ChromaDB memory could adopt typed memory items (profile, event, knowledge, behavior, skill, tool).

#### DAG Workflow Orchestration
- **Finding:** Dify uses a directed acyclic graph for workflow orchestration with NodeFactory protocol, bidirectional edge traversal, error strategy promotion, and branch state management.
- **Source:** langgenius/dify (`api/core/workflow/graph/graph.py`)
- **Relevance:** Complex multi-step Animus tasks could be modeled as DAGs for dependency-aware execution.

---

### Goal 2: Module Organization Patterns

#### Feature-Based Organization (Consensus Pattern)
All analyzed repos converge on feature-based organization with self-contained modules:

| Repository | Organization Style |
|------------|-------------------|
| gpt-engineer | `core/` (ABCs + defaults), `applications/` (CLI), `preprompts/` (templates) |
| rho | `extensions/{name}/` per feature, `skills/{name}/SKILL.md` |
| VoltAgent | `packages/core/src/{domain}/` (agent, tool, workflow, memory, mcp) |
| Rich | One module per renderable type (`table.py`, `panel.py`, `progress.py`) |
| yt-dlp | `extractor/` (1000+), `postprocessor/`, `plugins.py`, `utils/` |
| Dify | Deep nesting: `api/core/{domain}/{subdomain}/` |
| Claude-Mem | Domain-driven: `services/{domain}/{subdomain}/` |
| Browser-Use | `agent/`, `browser/`, `llm/`, `tools/`, `skills/`, `dom/` |

**Recommended structure for Animus:**
```
animus/
  core/          - Agent base, Runnable protocol, config, exceptions
  models/        - Model abstraction layer
    providers/   - llama_cpp.py, ollama.py, openai_compat.py
  tools/         - Tool registry, decorator-based registration
  skills/        - Skill loading, SKILL.md parsing, execution
    definitions/ - SKILL.md files with YAML frontmatter
  memory/        - Progressive disclosure memory
    stores/      - SQLite, ChromaDB backends
    search/      - Hybrid search orchestrator
  agents/        - Agent loop, sub-agent orchestration
    prompts/     - System prompts per capability tier
  rag/           - Retrieval pipeline, chunking, indexing
  mcp/           - MCP client/server
  cli/           - Typer CLI, Rich output
```

#### Skill-as-Directory Pattern
- **Finding:** Superpowers and awesome-claude-skills both use a directory-per-skill pattern with standardized files.
- **Source:** obra/superpowers, ComposioHQ/awesome-claude-skills
- **Relevance:** Each Animus skill should be:
  ```
  skill-name/
    SKILL.md       (required) - YAML frontmatter + instructions
    scripts/       (optional) - Executable code for deterministic tasks
    references/    (optional) - Documentation loaded on demand
    assets/        (optional) - Templates, config files
  ```

#### Three-Tier Monorepo (LangChain)
- **Finding:** LangChain separates into `langchain_core` (interfaces), `langchain` (chains/agents), `langchain_community` (integrations).
- **Source:** langchain-ai/langchain
- **Relevance:** Animus could separate core interfaces from community contributions as the project scales.

---

### Goal 3: Dependency Injection Usage

#### Constructor Injection with Sensible Defaults (Most Common)
- **Finding:** gpt-engineer's `SimpleAgent.__init__` takes all dependencies as optional parameters with factory defaults plus a `with_default_config` classmethod factory.
- **Source:** AntonOsika/gpt-engineer (`gpt_engineer/core/default/simple_agent.py`)
- **Code Example:**
  ```python
  class SimpleAgent(BaseAgent):
      def __init__(self, memory: BaseMemory, execution_env: BaseExecutionEnv,
                   ai: AI = None, preprompts_holder: PrepromptsHolder = None):
          self.ai = ai or AI()
          self.preprompts_holder = preprompts_holder or PrepromptsHolder(PREPROMPTS_PATH)

      @classmethod
      def with_default_config(cls, path, ai=None, ...):
          return cls(memory=DiskMemory(memory_path(path)), ...)
  ```

#### Protocol-Based DI (Structural Subtyping)
- **Finding:** memU uses Python `Protocol` with `@runtime_checkable` for database backends — structural subtyping without explicit inheritance.
- **Source:** NevaMind-AI/memU (`src/memu/database/interfaces.py`)
- **Relevance:** Animus could define tool/skill/model interfaces as Protocols, enabling third-party implementations without requiring base class inheritance.

#### Weakref-Based Bidirectional Linking
- **Finding:** agent-lightning uses weakrefs for Agent↔Trainer↔Runner links to prevent circular reference issues.
- **Source:** microsoft/agent-lightning (`agentlightning/litagent/litagent.py`)
- **Code Example:**
  ```python
  class LitAgent(Generic[T]):
      def set_trainer(self, trainer: Trainer) -> None:
          self._trainer_ref = weakref.ref(trainer)
  ```

#### Override-Deps Pattern
- **Finding:** Repomix passes `overrideDeps` parameters to every pipeline function for testing.
- **Source:** yamadashy/repomix
- **Code Example:**
  ```typescript
  const deps = { ...defaultDeps, ...overrideDeps }
  ```

#### Config-as-Context Threading
- **Finding:** LangChain threads `RunnableConfig` through every `invoke()` call for runtime parameter injection.
- **Source:** langchain-ai/langchain (`langchain_core/runnables/`)

**Recommended DI strategy for Animus:**
1. Constructor injection for services (memU pattern)
2. Protocol-based interfaces for all extension points
3. Factory pattern for model providers (Dify/WrenAI pattern)
4. Config threading through pipeline stages (LangChain pattern)

---

### Goal 4: Error Handling Strategies

#### Unified Error Translation (5 Canonical Types)
- **Finding:** Dify normalizes all model provider errors into 5 canonical types via `_invoke_error_mapping`. Consumer code handles only these 5 types regardless of backend.
- **Source:** langgenius/dify (`api/core/model_runtime/errors/invoke.py`)
- **Relevance:** Animus should map all llama-cpp-python, Ollama, and API errors to canonical types.
- **Code Example:**
  ```python
  class InvokeError(ValueError): ...
  class InvokeConnectionError(InvokeError): ...
  class InvokeRateLimitError(InvokeError): ...
  class InvokeAuthorizationError(InvokeError): ...
  class InvokeServerUnavailableError(InvokeError): ...
  class InvokeBadRequestError(InvokeError): ...
  ```

#### Parse-Error Self-Correction
- **Finding:** LangChain's `OutputParserException` has a `send_to_llm=True` flag that sends the error and offending output back to the model for self-correction. Combined with `.with_retry()`, this creates a robust parse-retry-correct loop.
- **Source:** langchain-ai/langchain (`langchain_core/exceptions.py`)
- **Relevance:** Critical for Animus — small local models frequently produce malformed output. Feed the error back and retry.

#### Expected vs Unexpected Classification
- **Finding:** yt-dlp's error hierarchy has an `expected` flag that determines whether to show "bug report" messages. Expected errors trigger retry/skip; unexpected errors suggest filing issues.
- **Source:** yt-dlp/yt-dlp (`yt_dlp/utils/_utils.py`)
- **Relevance:** Animus agent errors should be classified: expected (model produced bad JSON → retry) vs unexpected (segfault → report).

#### AllVariantsFailed Diagnostics
- **Finding:** TensorZero collects all variant errors in an `IndexMap<String, Error>` — when everything fails, the diagnostic includes every attempt's error.
- **Source:** tensorzero/tensorzero (`tensorzero-core/src/endpoints/inference.rs`)

#### Consecutive Failure Escalation
- **Finding:** Browser-Use tracks `consecutive_failures` with escalating response: retry → nudge with clarification → fallback LLM → stop.
- **Source:** browser-use/browser-use (`browser_use/agent/`)

#### Graceful Degradation
- **Finding:** Repomix's security filtering returns "suspicious" collections rather than hard failures, enabling partial success. yt-dlp's `--ignore-errors` lets batch operations continue past individual failures.
- **Source:** yamadashy/repomix, yt-dlp/yt-dlp

#### JSON-Serializable Error Hierarchy
- **Finding:** ChatDev defines a base `MACException` with `to_dict()` and `to_json()` methods, plus domain-specific subclasses (ValidationError, SecurityError, WorkflowExecutionError, TimeoutError, ExternalServiceError).
- **Source:** OpenBMB/ChatDev (`utils/exceptions.py`)
- **Relevance:** Errors that serialize to JSON are consumable by the model for self-correction and by logging systems for analysis.

---

### Goal 5: Features That Would Enhance Animus

#### Immediate Priority

1. **GBNF Grammar-Constrained Decoding** (llama.cpp)
   - Force model output to conform to a formal grammar, eliminating structural failures
   - JSON Schema → GBNF conversion automates constraint generation
   - llama-cpp-python already supports the `grammar` parameter
   - **Impact:** Eliminates malformed tool calls, the #1 failure mode with small models

2. **Progressive Disclosure Memory** (Claude-Mem)
   - Three-layer retrieval: search returns compact index (~50-100 tokens per result) → timeline shows context → get_observations fetches full details
   - Achieves ~10x token savings vs. loading all memory
   - Granular vector indexing: each fact/concept is a separate ChromaDB document
   - **Impact:** Makes RAG usable within small model context windows

3. **Variant/Fallback System** (TensorZero, Browser-Use)
   - Multiple strategies per task with automatic fallback
   - Best-of-N sampling: generate multiple outputs, select best via judge
   - Model switching mid-execution on consecutive failures
   - **Impact:** Layers reliability around unreliable small models

4. **Parse-Retry-Correct Loop** (LangChain)
   - `OutputParserException(send_to_llm=True)` feeds errors back to the model
   - Combined with `.with_retry()` for automatic recovery
   - **Impact:** Handles the inevitable malformed outputs from small models

5. **Capability-Tiered System Prompts** (Browser-Use)
   - Separate `system_prompt_flash.md` and `system_prompt_no_thinking.md` for smaller models
   - `max_actions_per_step=1` to limit hallucination in multi-action sequences
   - **Impact:** Different prompts for different model capabilities

#### Near-Term Priority

6. **Skill-as-Markdown with Scaffolding** (Superpowers, awesome-claude-skills)
   - SKILL.md with YAML frontmatter for human-editable skill definitions
   - `init_skill.py` for scaffolding, `package_skill.py` for validation
   - Progressive disclosure: metadata (~100 words) → body (<5k words) → resources (unlimited)
   - Scripts for deterministic tasks (execute without loading into context)

7. **Decorator-Based Tool Registry** (Browser-Use)
   - `@registry.action(description, param_model)` registration
   - Dynamic Pydantic Union generation for LLM tool calling schema
   - Domain filtering restricts available tools by context
   - **Impact:** Adding a tool requires only a function + decorator

8. **Sub-Agent Orchestration with Bail** (VoltAgent, Maestro)
   - Sub-agents can signal "I have the final answer" to short-circuit supervisor loop
   - Fresh context per sub-agent (no context pollution)
   - Parallel dispatch for independent tasks, sequential for dependent ones

9. **Knowledge Compounding Loop** (compound-engineering-plugin)
   - Build → Solve → Document → Search → Accelerate cycle
   - `docs/solutions/` indexed by RAG for accelerating future tasks
   - Each solved problem makes subsequent work easier

10. **Code-as-Action Mode** (pydantic/monty)
    - Agent writes Python using tool functions; execute in sandbox
    - Collapses N tool-call round trips into 1 code-generation step
    - Snapshot/resume enables human-in-the-loop approval

#### Strategic Priority

11. **Feedback Flywheel** (TensorZero, agent-lightning)
    - Store every inference with structured telemetry
    - Collect feedback (automated validators + human corrections)
    - Use traces to optimize prompts via RL/prompt optimization
    - `emit_xxx()` pattern for non-invasive telemetry

12. **LoRA Fine-Tuning Pipeline** (rasbt/LLMs-from-scratch)
    - Instruction fine-tuning with masked loss (only on response tokens)
    - Per-task LoRA adapters (code generation, code review, planning)
    - DPO for aligning models using production success/failure data

13. **Action Loop Detector** (Browser-Use)
    - Rolling window of action hashes with escalating nudges at thresholds
    - Page/state fingerprints via SHA-256
    - Essential for preventing infinite loops in agent execution

14. **Hybrid RAG with Parallel Retrieval** (Dify, Claude-Mem)
    - Keyword + vector + full-text search in parallel with deduplication
    - Strategy-based fallback (ChromaDB → SQLite)
    - Model-based reranking for precision

---

### Goal 6: Making Smaller Models Do Productive Agentic Work

This is the synthesis of the most critical findings across all 91 repositories.

#### Strategy 1: Constrain the Output Space
- **GBNF Grammar-Constrained Decoding** (llama.cpp): Force valid JSON, valid tool calls, valid code syntax at the token level. The model's capacity is freed from "formatting duty" and focused on content.
- **Structured Output Schemas** (codex-action, VoltAgent, Browser-Use): Define exact output shapes with Pydantic/JSON Schema. Reject and retry on schema violations.
- **Prompt Templates with Conditional Sections** (WrenAI): Only include relevant context, keeping prompts focused for small context windows.

#### Strategy 2: Decompose Tasks to Atomic Units
- **2-5 Minute Task Units** (Superpowers): Each task includes exact file paths, code snippets, and CLI commands — reducing the model's need to "figure things out."
- **Fresh Context Per Task** (Superpowers, Maestro): No accumulated context pollution. Each sub-agent gets exactly what it needs.
- **Single Action Per Step** (Browser-Use): `max_actions_per_step=1` prevents multi-action hallucination.

#### Strategy 3: Layer Reliability Around Unreliable Components
- **Variant Fallback** (TensorZero): Try multiple strategies; first success wins.
- **Best-of-N Sampling** (TensorZero): Generate multiple outputs, judge and select the best.
- **Parse-Retry-Correct Loops** (LangChain): Catch parse failures, show error to model, retry.
- **Consecutive Failure Escalation** (Browser-Use): retry → nudge → fallback model → stop.

#### Strategy 4: Provide Rich Context via Memory and RAG
- **Progressive Disclosure** (Claude-Mem, Superpowers): Index → filter → fetch to stay within small context windows.
- **DICL — Dynamic In-Context Learning** (TensorZero): Retrieve successful past interactions as few-shot examples at inference time.
- **Knowledge Compounding** (compound-engineering): Index past solutions for future reference.
- **Granular Vector Indexing** (Claude-Mem): Fact-level vectors for precise retrieval.

#### Strategy 5: Use Tiered Models for Different Tasks
- **Separate Models for Different Cognitive Tasks** (rho, awesome-llm-apps): Use main model for reasoning, smaller/faster model for memory extraction and summarization.
- **Capability-Tiered Prompts** (Browser-Use): Simpler, more prescriptive prompts for smaller models.
- **CoT-Based Tool Calling** (Dify): Parse tool invocations from free-text for models without native function calling.

#### Strategy 6: Augment with Tools
- **Token Forcing** (nanochat): Inject computed results directly into the generation stream.
- **Scripts for Deterministic Tasks** (awesome-claude-skills): Execute without loading into context — saves tokens for reasoning.
- **Calculator/Code Execution** (nanochat): Offload computation the model cannot do internally.

#### Strategy 7: Controller-as-Curator
- **Exactly What Context Is Needed** (Superpowers): The orchestrator provides focused context rather than making sub-agents search.
- **Bidirectional Q&A Before Implementation** (Superpowers): Force a clarification cycle — small models make fewer mistakes when they can ask questions first.
- **Two-Stage Review Gates** (Superpowers): Spec review then code quality review catches errors via multiple passes.

#### Strategy 8: Invest in Long-Term Improvement
- **LoRA Adapters** (rasbt): Cheap per-task specialization without full model retraining.
- **Feedback Flywheel** (TensorZero): Store outcomes, learn from them, improve prompts over time.
- **Synthetic Training Data** (nanochat): Targeted synthetic data teaches specific capabilities.

---

## Cross-Cutting Themes

### Theme 1: Skills Systems Are Converging on Markdown + YAML Frontmatter
Every skills-oriented repository (Superpowers, awesome-claude-skills, pi-skills, oh-my-opencode, compound-engineering) uses SKILL.md files with YAML frontmatter. This is becoming a de facto standard. Animus should fully embrace this format.

### Theme 2: Memory Is the Differentiator
The repos that produce the most capable agents (rho, Claude-Mem, memU) all invest heavily in persistent, structured memory. Raw chat history is insufficient — memory needs to be typed (profile, event, knowledge, behavior, skill, tool), deduplicated, decayed, and progressively disclosed.

### Theme 3: Grammar Constraints Are the Single Highest-Impact Change for Local Models
llama.cpp's GBNF grammar system eliminates the entire class of "structurally invalid output" failures. This is more impactful than any prompt engineering technique because it operates at the token level — the model literally cannot produce invalid JSON or malformed tool calls.

### Theme 4: Reliability Comes from Layers, Not Better Models
TensorZero, Browser-Use, VoltAgent, and LangChain all demonstrate that wrapping unreliable components in fallback chains, retry logic, parse-correction loops, and validation gates produces reliable systems from unreliable parts. This is the core strategy for local model agents.

### Theme 5: Code-as-Action Is an Underexplored Efficiency Gain
Monty (pydantic) and nanochat both show that having the model write Python that calls tool functions (executed in a sandbox) collapses multiple tool-call round trips into a single inference. This reduces the number of model calls, which is the primary bottleneck with local inference.

---

## Recommendations

### Immediate (Next Sprint)

1. **Implement GBNF grammar constraints** for all tool call schemas in Animus's llama-cpp-python integration. Define grammars for each tool's argument schema and use the `grammar` parameter.

2. **Add parse-retry-correct loop** to the agent's output parsing. When JSON parsing fails, include the error and offending output in the retry prompt. Limit to 3 retries.

3. **Implement progressive disclosure for RAG results**. Return compact indices first (title, score, ID), then expand selected results. Budget tokens explicitly.

4. **Create capability-tiered system prompts**. Define `system_prompt_full.md`, `system_prompt_compact.md`, and `system_prompt_minimal.md` for different model sizes.

### Near-Term (Next 2-3 Sprints)

5. **Refactor skills to SKILL.md format** with YAML frontmatter, optional scripts/, references/, assets/ directories. Add `animus skill create <name>` scaffolding command.

6. **Implement model fallback chain**: local small → local large → API (if configured). Use consecutive failure counting with escalation.

7. **Add decorator-based tool registry**: `@tool(description, param_model)` registration with dynamic schema generation.

8. **Implement knowledge compounding**: After each successful task, store the solution in `docs/solutions/` with structured metadata for RAG indexing.

9. **Add action loop detection**: Monitor for repetitive agent behavior with escalating intervention.

### Strategic (Backlog)

10. **Build feedback flywheel infrastructure**: Log all inferences with structured telemetry. Collect automated validator feedback (did the JSON parse? did the code compile?). Use for prompt optimization.

11. **Implement code-as-action mode**: Allow the agent to write Python using tool functions, execute in a sandboxed interpreter. Reduces inference count for multi-step tasks.

12. **Add LoRA fine-tuning pipeline**: Collect successful tool call sequences as training data. Fine-tune per-task adapters.

13. **Implement DICL (Dynamic In-Context Learning)**: Retrieve relevant past successful interactions and inject as few-shot examples at inference time.

14. **Add unified error translation layer**: Map all backend-specific errors to 5 canonical types (Connection, Model, RateLimit, Auth, BadRequest).

---

## Per-Repo Summary Table

### Previously Analyzed (42 repos from tasks.md)

| Repository | Key Patterns Borrowed | Hardcoded % |
|------------|----------------------|-------------|
| potpie-ai/potpie | Knowledge graph for code (Neo4j), specialist agents | — |
| firecrawl/firecrawl | Multi-format output, async job queuing, anti-bot handling | — |
| bluewave-labs/Checkmate | Distributed job queues, centralized error middleware | — |
| openclaw/openclaw | WebSocket gateway, tool policies, sandbox execution | — |
| pranshuparmar/witr | Cross-platform CLI patterns, hierarchical detection | — |
| anomalyco/opencode | Three-tier permissions, MCP/LSP integration, explore/plan agents | — |
| memvid/memvid | Smart Frames, hybrid search (BM25+vector), time-travel debugging | — |
| C4illin/ConvertX | Modular converter architecture, 22 integrated tools | — |
| itsOwen/CyberScraper-2077 | LLM-based scraping, stealth mode, multi-format export | — |
| browseros-ai/BrowserOS | Browser as MCP server (31 tools), local-first agents | — |
| metorial/metorial | 600+ MCP integrations, OAuth session management | — |
| eigent-ai/eigent | Multi-agent workforce, dynamic task decomposition | — |
| charmbracelet/crush | LSP integration, MCP support, Agent Skills Standard | — |
| yashab-cyber/nmap-ai | AI-powered scanning, natural language interface | — |
| lemonade-sdk/lemonade | OpenAI-compatible local API, multi-backend (GGUF/ONNX) | — |
| yashab-cyber/metasploit-ai | Intelligent exploit ranking, multi-interface | — |
| assafelovic/gpt-researcher | Planner-executor pattern, parallel crawlers | — |
| jesseduffield/lazygit | Undo/redo with reflog, line-level staging | — |
| janhq/jan | OpenAI-compatible local API, MCP, extension system | — |
| QwenLM/Qwen3-Coder | 256K-1M context, 358 languages, FIM, custom tool parser | — |
| FlowiseAI/Flowise | Visual agent builder, multi-agent collaboration, RAG | — |
| aquasecurity/tracee | eBPF runtime security, behavioral detection | — |
| n8n-io/n8n | 400+ integrations, hybrid code/visual, LangChain AI | — |
| logpai/loghub | 16+ log datasets, AI-driven log analytics research | — |
| anthropics/claude-code-action | Context-aware GitHub Action, progress tracking | — |
| lizTheDeveloper/ai_village | ECS architecture (211 systems), multiverse forking | — |
| browseros-ai/moltyflow | Agent-to-agent Q&A, karma system | — |
| Legato666/katana | Dual-mode crawling, scope control, resume capability | — |
| anthropic-experimental/sandbox-runtime | OS-level sandboxing, mandatory deny lists | 100% |
| VoidenHQ/voiden | Hook registry with priorities, IPC tool system | 95% |
| automazeio/ccpm | File-based context, parallel agent coordination | 90% |
| mitsuhiko/agent-stuff | Event-driven lifecycle, TUI components, fuzzy matching | 85% |
| badlogic/pi-skills | SKILL.md format, YAML frontmatter, CLI tool abstraction | 100% |
| nicobailon/pi-subagents | Chain execution, fan-out/fan-in, async jobs | 80% |
| supermemoryai/supermemory | Normalized embeddings, multi-tier fallback, relevance scoring | 70% |
| assafelovic/skyll | Protocol-based sources, relevance ranking, LRU caching | 90% |
| oxidecomputer/dropshot | Type-safe extractors, trait-based APIs, OpenAPI generation | 100% |
| adenhq/hive | Node graphs, edge conditions, semantic failure detection | 60% |
| anthropics/claude-code | Agentic CLI, codebase understanding, git workflow | — |
| anthropics/skills | SKILL.md format, dynamic capability extension | — |
| anthropics/claude-cookbooks | RAG patterns, tool use, sub-agents, vision | — |
| stakpak/agent | Rust MCP, secret substitution, mTLS | 100% |

### Newly Analyzed (42 repos)

| Repository | Key Patterns Found | Relevance to Animus |
|------------|-------------------|---------------------|
| **AntonOsika/gpt-engineer** | Step-based pipeline, ABC+defaults, preprompts system, FilesDict, chat-to-files parser | HIGH — Agent loop architecture, prompt template system |
| **microsoft/agent-lightning** | LightningStore mediator, emit_xxx() telemetry, Agent-Runner-Trainer triangle, weakref DI | HIGH — Telemetry, prompt optimization loop |
| **mikeyobrien/rho** | Extension-based architecture, JSONL persistent memory, memory decay, auto-memory extraction, tiered model strategy | HIGH — Memory persistence, skill-as-markdown |
| **op7418/CodePilot** | Promise-based permission registry, SSE streaming, SQLite session persistence, MCP transport abstraction | MEDIUM — Permission gates, session storage |
| **pedramamini/Maestro** | Manager-based architecture, discriminated union configs, three-tier message routing, git worktree orchestration, playbook system | HIGH — Sub-agent coordination, task isolation |
| **openai/codex-action** | Pipeline with safety layers, structured output schema validation, strict input parsing, stdin prompt delivery | HIGH — Output schema enforcement, safety strategies |
| **pydantic/monty** | Interpreter-as-sandbox, snapshot/resume execution, resource limiting builder pattern, external functions as tools | HIGH — Code-as-action mode, sandboxed execution |
| **VoltAgent/voltagent** | Registry-based plugin architecture, manager delegation, middleware pipeline with retry, guardrails (allow/block/modify), model fallback chain, error serialization for models | HIGH — Middleware, guardrails, fallback |
| **openai/skills** | Playwright browser automation skills, bash CLI tools, structured SKILL.md format | MEDIUM — Skill format reference |
| **OpenBMB/ChatDev** | Multi-agent software development, role-based agents, JSON-serializable error hierarchy | MEDIUM — Error patterns |
| **thedotmack/claude-mem** | Progressive disclosure memory, MCP thin wrapper, hybrid search (SQLite FTS + ChromaDB), granular vector indexing, context builder pipeline, token economics | HIGH — Memory architecture, RAG strategy |
| **code-yeongyu/oh-my-opencode** | Claude Code extension framework, Sisyphus orchestrator, task management integration, skill MCP manager, tmux sub-agents | MEDIUM — Extension patterns |
| **ggml-org/llama.cpp** | GBNF grammar-constrained decoding, composable sampling pipeline, ring buffer history, JSON Schema → GBNF conversion | CRITICAL — Grammar constraints for tool calls |
| **karpathy/nanochat** | Single-dial scaling, tool use via token forcing, safe evaluation with timeout, differentiated optimizer groups | HIGH — Tool augmentation for small models |
| **rasbt/LLMs-from-scratch** | Instruction fine-tuning pipeline, LoRA adapters, masked loss (ignore_index=-100), DPO alignment | HIGH — Fine-tuning infrastructure |
| **tensorzero/tensorzero** | Gateway pattern, variant-based experimentation, Best-of-N/DICL/Mixture-of-N, feedback flywheel, AllVariantsFailed diagnostics | CRITICAL — Reliability layers, optimization |
| **NevaMind-AI/memU** | Three-layer memory hierarchy, Protocol-based DI, repository pattern, factory with lazy imports, workflow pipeline with immutable revisions, interceptor pattern, content-hash dedup | HIGH — Memory architecture, DI patterns |
| **KeygraphHQ/shannon** | Narrow agent scoping, parallel sub-agent execution, Promise.allSettled pattern | MEDIUM — Sub-agent patterns |
| **langchain-ai/langchain** | Runnable protocol, pipe composition, .with_retry()/.with_fallbacks(), OutputParserException(send_to_llm=True), callback mixin system, RunnableWithMessageHistory | HIGH — Pipeline composition, error recovery |
| **langgenius/dify** | Model provider factory with plugin discovery, unified invoke error hierarchy (5 types), DAG workflow engine, hybrid RAG with parallel retrieval, CoT-based tool calling | HIGH — Error handling, RAG, workflow |
| **browser-use/browser-use** | Step-based agent loop, decorator-based tool registry, dynamic Pydantic Union generation, action loop detector, capability-tiered prompts, fallback LLM switching, message compaction | HIGH — Tool registry, loop detection, prompts |
| **Textualize/rich** | Protocol-based polymorphism, recursive rendering pipeline, render hooks, Live display with threading, Progress with rolling-window estimation | MEDIUM — Already used; reference for advanced patterns |
| **yamadashy/repomix** | Sequential pipeline with DI, hierarchical config, Tree-sitter code compression, security scanning, XML output for LLM parsing, token counting per file | HIGH — Context packing, token management |
| **yt-dlp/yt-dlp** | Extractor pattern (auto-routing via _VALID_URL), importlib plugin discovery, expected/unexpected error classification, lazy module loading, graceful degradation | HIGH — Plugin architecture, error handling |
| **obra/superpowers** | SKILL.md with progressive disclosure, mandatory skill checking, skill shadowing, sub-agent orchestration with role specialization, task decomposition to 2-5 min units, bidirectional Q&A, two-stage review gates | CRITICAL — Skills, task decomposition, small model productivity |
| **ComposioHQ/awesome-claude-skills** | Skill template + scaffolding, scripts for deterministic reliability, references with grep patterns, validation/packaging pipeline, meta-skill pattern | HIGH — Skill ecosystem patterns |
| **Canner/WrenAI** | Pipeline RAG architecture, @provider decorator + auto-discovery, PipelineComponent as DI container, Jinja2 prompt templates with conditional blocks, semantic layer for output constraining | HIGH — Provider pattern, prompt templates |
| **EveryInc/compound-engineering-plugin** | Parallel sub-agent orchestration, knowledge compounding loop, #$ARGUMENTS template variables, agent specialization taxonomy, resolve-parallel pattern | HIGH — Knowledge compounding, orchestration |
| **opendatalab/MinerU** | Singleton model cache (config-keyed), multi-backend strategy, batch processing pipeline, hardware-aware model selection, per-item error recovery | MEDIUM — Model caching, batch processing |
| **Shubhamsaboo/awesome-llm-apps** | DeepSeek local RAG agent, fallback chain (RAG→web→direct), similarity threshold tuning, thinking process extraction, separate models for different tasks | HIGH — Local RAG patterns |
| **TibixDev/winboat** | Simple agent wrapper pattern | LOW |
| **badlogic/pi-mono** | Monorepo with npm workspaces, 7 packages, AGENTS.md for AI contributors | LOW — Monorepo reference |
| **laude-institute/terminal-bench** | CLI agent benchmarking, ~100 tasks, containerized sandbox | MEDIUM — Evaluation framework |
| **ThePrimeagen/99** | Neovim AI plugin, # syntax for skills, @ for file references, multi-agent support | LOW |
| **CadQuery/cadquery** | Fluent/chainable API, parametric design, workplane-based execution model | LOW — API design reference |
| **scikit-learn/scikit-learn** | Estimator pattern (fit/predict), pipeline composition, consistent interface across algorithms | MEDIUM — Pipeline composition pattern |
| **AliKarasneh/automatisch** | Trigger-action workflows, data sovereignty, dual licensing (AGPL + enterprise) | LOW |
| **govctl-org/govctl** | RFC-driven governance, phase-gated workflow (SPEC→IMPL→TEST→STABLE), policy-as-code | MEDIUM — Governance patterns |
| **iOfficeAI/AionUi** | Multi-agent mode, persistent sessions, markdown skill system, scheduled tasks | LOW — GUI reference |
| **tobi/qmd** | BM25+vector hybrid search with Reciprocal Rank Fusion (RRF), 800-token chunks with 15% overlap, path-aware context | MEDIUM — Search/RAG patterns |
| **saeidrastak/local-ai-packaged** | Docker Compose AI stack (n8n + Ollama + Qdrant + Neo4j + Flowise + SearXNG), modular GPU profiles | LOW — Deployment reference |
| **justlovemaki/AIClient-2-API** | Strategy+adapter pattern for multi-provider API, provider pool management, protocol translation layer | LOW — API abstraction reference |

---

## Next Steps

1. **Phase 18: Grammar-Constrained Tool Calling** — Implement GBNF grammars for all existing tool schemas. Add JSON Schema → GBNF conversion. Test with Qwen3-VL and other local models.

2. **Phase 19: Progressive Disclosure Memory** — Refactor RAG to return compact indices first. Add token budget management. Implement granular vector indexing (fact-level).

3. **Phase 20: Reliability Layers** — Add parse-retry-correct loop, variant/fallback system, consecutive failure escalation, and action loop detection to the agent loop.

4. **Skill System Enhancement** — Migrate to SKILL.md format with YAML frontmatter. Add scaffolding command. Implement progressive disclosure (metadata → body → resources).

5. **Evaluation Infrastructure** — Set up terminal-bench-style evaluation for measuring local model productivity improvements. Track tool call success rate, parse failure rate, and task completion rate across model sizes.

6. **Benchmark Before/After** — Measure current tool call success rate with Qwen3-VL, then re-measure after implementing grammar constraints and parse-retry loops. Target: >95% structurally valid outputs.
