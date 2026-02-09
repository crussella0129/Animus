# GECK Repor Agent Instructions

## Project Information

- **Project Name:** Animus
- **Working Directory:** /home/charles/Git_Repos/Animus
- **GECK Folder:** /home/charles/Git_Repos/Animus/GECK
- **Project Git Repo:** git@github.com:crussella0129/Animus.git

## Repositories to Explore

- https://github.com/potpie-ai/potpie
- https://github.com/firecrawl/firecrawl
- https://github.com/bluewave-labs/Checkmate
- https://github.com/openclaw/openclaw
- https://github.com/pranshuparmar/witr
- https://github.com/anomalyco/opencode
- https://github.com/memvid/memvid
- https://github.com/C4illin/ConvertX
- https://github.com/itsOwen/CyberScraper-2077
- https://github.com/Legato666/katana
- https://github.com/browseros-ai/BrowserOS
- https://github.com/metorial/metorial
- https://github.com/TibixDev/winboat
- https://github.com/eigent-ai/eigent
- https://github.com/charmbracelet/crush
- https://github.com/yashab-cyber/nmap-ai
- https://github.com/lemonade-sdk/lemonade
- https://github.com/yashab-cyber/metasploit-ai
- https://github.com/assafelovic/gpt-researcher
- https://github.com/jesseduffield/lazygit
- https://github.com/janhq/jan
- https://github.com/QwenLM/Qwen3-Coder
- https://github.com/FlowiseAI/Flowise
- https://github.com/aquasecurity/tracee
- https://github.com/n8n-io/n8n
- https://github.com/logpai/loghub
- https://github.com/anthropics/claude-code-action
- https://github.com/lizTheDeveloper/ai_village
- https://github.com/browseros-ai/moltyflow
- https://github.com/anthropic-experimental/sandbox-runtime
- https://github.com/VoidenHQ/voiden
- https://github.com/automazeio/ccpm
- https://github.com/badlogic/pi-mono
- https://github.com/mitsuhiko/agent-stuff
- https://github.com/tensorzero/tensorzero
- https://github.com/laude-institute/terminal-bench
- https://github.com/badlogic/pi-skills
- https://github.com/nicobailon/pi-subagents
- https://github.com/yamadashy/repomix
- https://github.com/CadQuery/cadquery
- https://github.com/adenhq/hive
- https://github.com/assafelovic/skyll
- https://github.com/scikit-learn/scikit-learn
- https://github.com/oxidecomputer/dropshot
- https://github.com/supermemoryai/supermemory
- https://github.com/VoltAgent/voltagent
- https://github.com/AliKarasneh/automatisch
- https://github.com/pedramamini/Maestro
- https://github.com/ThePrimeagen/99
- https://github.com/thedotmack/claude-mem
- https://github.com/microsoft/agent-lightning
- https://github.com/govctl-org/govctl
- https://github.com/openai/codex-action
- https://github.com/ggml-org/llama.cpp
- https://github.com/saeidrastak/local-ai-packaged
- https://github.com/justlovemaki/AIClient-2-API
- https://github.com/KeygraphHQ/shannon
- https://github.com/yt-dlp/yt-dlp
- https://github.com/langgenius/dify
- https://github.com/langchain-ai/langchain
- https://github.com/rasbt/LLMs-from-scratch
- https://github.com/Shubhamsaboo/awesome-llm-apps
- https://github.com/browser-use/browser-use
- https://github.com/Textualize/rich
- https://github.com/AntonOsika/gpt-engineer
- https://github.com/opendatalab/MinerU
- https://github.com/mikeyobrien/rho
- https://github.com/op7418/CodePilot
- https://github.com/pydantic/monty
- https://github.com/EveryInc/compound-engineering-plugin
- https://github.com/iOfficeAI/AionUi
- https://github.com/NevaMind-AI/memU
- https://github.com/code-yeongyu/oh-my-opencode
- https://github.com/ComposioHQ/awesome-claude-skills
- https://github.com/openai/skills
- https://github.com/obra/superpowers
- https://github.com/tobi/qmd
- https://github.com/Canner/WrenAI
- https://github.com/OpenBMB/ChatDev
- https://github.com/karpathy/nanochat

## Exploration Goals

- Identify architectural patterns (MVC, etc.)
- Find module organization patterns
- Locate dependency injection usage
- Discover error handling strategies
- Find features from each of these repos that would enhance Animus
- Of those associated with CLI agents, with locally running models, see how they solve the challenge of getting smaller models to do productive agentic work.

## Instructions

You are an AI exploration agent tasked with analyzing external repositories to find improvements, patterns, and ideas that can be applied to the project above.

### Your Mission

1. **Clone and Explore** each repository listed above
2. **Search for** implementations, patterns, and techniques related to the exploration goals
3. **Document Findings** in the GECK folder with:
   - Code snippets that demonstrate useful patterns
   - Links to specific files/lines in the source repos
   - Explanations of how each finding could apply to this project

### Output Format

Create a file `GECK/repor_findings.md` with:

```markdown
# Repor Findings — Animus

**Generated:** [timestamp]

## Summary
[Brief overview of what was found]

## Findings by Goal

### [Goal 1]
- **Finding:** [description]
- **Source:** [repo/file:line]
- **Relevance:** [how it applies to this project]
- **Code Example:**
  ```
  [relevant code snippet]
  ```

[Repeat for each finding]

## Recommendations
[Prioritized list of improvements to implement]

## Next Steps
[Suggested actions based on findings]
```

### Guidelines

- Focus on patterns that match the project's technology stack
- Prioritize findings that address the exploration goals
- Include enough context for each finding to be actionable
- Note any dependencies or prerequisites for implementing findings
- Flag any potential conflicts with existing project architecture