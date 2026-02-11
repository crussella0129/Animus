# Ornstein & Smough: Dual-Layer Container Isolation

**Status:** Design Phase
**Priority:** Medium
**Phase:** Container Isolation with `--paranoid` flag

## Concept

Named after the Dark Souls boss duo, the Ornstein & Smough system provides layered security for web exploration and untrusted code execution:

- **Ornstein (Dragonslayer)**: Fast, agile reconnaissance layer - lightweight sandboxing for web scraping and exploration
- **Smough (Executioner)**: Heavy enforcement layer - full container isolation for untrusted code execution

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    ANIMUS AGENT                         │
│                  (Trusted Context)                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ├─ Normal Mode (default)
                     │  └─> Direct execution (current behavior)
                     │
                     ├─ --cautious flag
                     │  └─> Ornstein Layer (lightweight sandbox)
                     │      • Process isolation
                     │      • Resource limits
                     │      • Network filtering
                     │      • Read-only filesystem mounts
                     │
                     └─ --paranoid flag
                        └─> Smough Layer (full container isolation)
                            └─> Ornstein (nested)
                                • Docker/Podman container
                                • No host access
                                • Ephemeral filesystem
                                • Network proxy with allowlist
                                • Syscall filtering (seccomp)
```

## Use Cases

### Ornstein (Lightweight Sandbox)
**When:** Web scraping, API calls, untrusted URL exploration
**Threats:** Malicious redirects, SSRF, resource exhaustion
**Mitigations:**
- Process-level isolation (subprocess with restricted privileges)
- Resource limits (CPU, memory, time)
- Network filtering (DNS allowlist, IP blocklist)
- Read-only filesystem mounts
- No shell access

### Smough (Heavy Container)
**When:** Running untrusted code, installing packages, browser automation
**Threats:** Code execution exploits, privilege escalation, host compromise
**Mitigations:**
- Full container isolation (Docker/Podman)
- Ephemeral filesystem (destroyed after execution)
- Network proxy with explicit allowlist
- Syscall filtering (seccomp profile)
- User namespace isolation
- No privileged operations

## Component Design

### 1. Ornstein Layer (`src/isolation/ornstein.py`)

```python
class OrnsteinSandbox:
    """Lightweight process-level sandbox for web exploration."""

    def __init__(self, config: OrnsteinConfig):
        self.cpu_limit: float = config.cpu_percent  # Max 50% CPU
        self.memory_limit: int = config.memory_mb    # Max 512 MB
        self.timeout: int = config.timeout_seconds   # Max 30s
        self.allowed_domains: list[str] = config.allowed_domains

    def execute_web_request(self, url: str) -> Response:
        """Execute web request in isolated subprocess."""
        # 1. Validate URL against allowlist/blocklist
        # 2. Spawn subprocess with resource limits
        # 3. Set network filtering (DNS resolver with allowlist)
        # 4. Execute request with timeout
        # 5. Return sanitized response

    def execute_script(self, script: str, language: str) -> Result:
        """Execute script in read-only environment."""
        # 1. Write script to temp directory
        # 2. Mount temp dir read-only
        # 3. Execute with interpreter (python, node, etc.)
        # 4. Capture stdout/stderr
        # 5. Cleanup temp directory
```

### 2. Smough Layer (`src/isolation/smough.py`)

```python
class SmoughContainer:
    """Heavy container isolation for untrusted execution."""

    def __init__(self, config: SmoughConfig):
        self.container_runtime = config.runtime  # docker or podman
        self.base_image = config.image           # alpine, ubuntu-minimal
        self.network_mode = "isolated"           # Custom bridge with proxy
        self.seccomp_profile = "strict"          # Block dangerous syscalls

    def create_ephemeral_container(self) -> Container:
        """Create disposable container with strict isolation."""
        # 1. Pull base image if not cached
        # 2. Create container with:
        #    - No privileged flags
        #    - Read-only root filesystem
        #    - Tmpfs for /tmp and /var
        #    - User namespace (non-root inside)
        #    - Seccomp profile (block ptrace, mount, etc.)
        #    - Network isolated to proxy
        # 3. Return container handle

    def execute_in_container(self, container: Container, command: list[str]) -> Result:
        """Execute command and return output."""
        # 1. Copy files to container if needed
        # 2. Execute command with timeout
        # 3. Stream output
        # 4. Cleanup and destroy container
```

### 3. Orchestration (`src/isolation/__init__.py`)

```python
def execute_with_isolation(
    action: Action,
    level: IsolationLevel = IsolationLevel.NONE
) -> Result:
    """
    Execute action with appropriate isolation level.

    Args:
        action: The action to execute (web request, script, command)
        level: NONE (direct), ORNSTEIN (lightweight), SMOUGH (heavy)

    Returns:
        Result with output, errors, and isolation metadata
    """
    if level == IsolationLevel.NONE:
        return execute_directly(action)
    elif level == IsolationLevel.ORNSTEIN:
        sandbox = OrnsteinSandbox.from_config()
        return sandbox.execute(action)
    elif level == IsolationLevel.SMOUGH:
        container = SmoughContainer.from_config()
        with container.ephemeral() as ctx:
            # Ornstein runs INSIDE Smough container
            return ctx.execute_with_ornstein(action)
```

## Configuration

### AnimusConfig Extension

```python
class IsolationConfig(BaseModel):
    """Container isolation configuration."""

    # Global settings
    default_level: str = "none"  # "none", "ornstein", "smough"

    # Ornstein (lightweight sandbox)
    ornstein_cpu_percent: float = 50.0
    ornstein_memory_mb: int = 512
    ornstein_timeout_seconds: int = 30
    ornstein_allowed_domains: list[str] = []  # Empty = all allowed
    ornstein_blocked_ips: list[str] = [
        "127.0.0.0/8",    # Localhost
        "10.0.0.0/8",     # Private
        "172.16.0.0/12",  # Private
        "192.168.0.0/16", # Private
    ]

    # Smough (heavy container)
    smough_runtime: str = "docker"  # "docker" or "podman"
    smough_base_image: str = "python:3.11-alpine"
    smough_network_proxy: str = ""  # Optional HTTP proxy
    smough_allowed_domains: list[str] = []  # Explicit allowlist
    smough_max_execution_time: int = 300  # 5 minutes
```

### CLI Flags

```bash
# Default: no isolation
animus rise

# Lightweight isolation for web exploration
animus rise --cautious

# Full container isolation for untrusted code
animus rise --paranoid

# Override default level for specific tools
animus rise --isolate shell=smough,web=ornstein
```

## Tool Integration

### Web Search/Scraping Tools

```python
@tool(isolation=IsolationLevel.ORNSTEIN)
def web_search(query: str) -> SearchResults:
    """Search the web (isolated by default)."""
    # Automatically routed through Ornstein sandbox

@tool(isolation=IsolationLevel.SMOUGH)
def web_scrape(url: str, extract: str) -> str:
    """Scrape and extract content from URL (heavy isolation)."""
    # Automatically routed through Smough container
```

### Shell Execution Tools

```python
@tool(isolation=IsolationLevel.NONE)
def shell_read(command: str) -> str:
    """Execute read-only shell command (direct)."""
    # Safe commands like ls, cat, grep

@tool(isolation=IsolationLevel.SMOUGH)
def shell_write(command: str) -> str:
    """Execute shell command with write access (isolated)."""
    # Dangerous commands like rm, wget, curl
```

## Security Properties

### Ornstein Layer Guarantees

✓ Process isolation (separate PID namespace)
✓ Resource limits prevent DoS
✓ Network filtering prevents SSRF
✓ Read-only mounts prevent tampering
✓ Timeout prevents infinite loops
✗ No protection against kernel exploits
✗ No protection against host escape

**Threat Model:** Malicious web content, resource exhaustion, SSRF attempts

### Smough Layer Guarantees

✓ Full container isolation (namespaces: PID, NET, MNT, IPC, UTS)
✓ Ephemeral filesystem (destroyed after execution)
✓ Network proxy with allowlist
✓ Seccomp filters block dangerous syscalls
✓ User namespace (non-root inside container)
✓ No privileged operations
✓ Protection against host escape
✗ Not VM-level isolation (shared kernel)

**Threat Model:** Untrusted code execution, zero-day exploits, privilege escalation

## Implementation Plan

### Phase 1: Ornstein Layer (Lightweight)
- [ ] `src/isolation/ornstein.py` - Process sandbox class
- [ ] Resource limiting (psutil)
- [ ] Network filtering (DNS resolver override)
- [ ] Read-only mounts
- [ ] Timeout enforcement
- [ ] Tests (20+)

### Phase 2: CLI Integration
- [ ] `--cautious` flag → Ornstein for web tools
- [ ] Config: `isolation.default_level`
- [ ] Tool decorator: `@tool(isolation=...)`
- [ ] Permission prompt for isolation level changes

### Phase 3: Smough Layer (Heavy)
- [ ] `src/isolation/smough.py` - Container management
- [ ] Docker/Podman detection and selection
- [ ] Ephemeral container creation
- [ ] Seccomp profile generation
- [ ] Network proxy setup
- [ ] Nested Ornstein execution
- [ ] Tests (30+)

### Phase 4: Full Integration
- [ ] `--paranoid` flag → Smough for all tools
- [ ] Per-tool isolation override
- [ ] Isolation level recommendation system
- [ ] Audit logging for isolated executions
- [ ] Documentation and examples

## Testing Strategy

### Unit Tests
- Ornstein resource limits enforcement
- Network filtering (allow/block)
- Timeout behavior
- Smough container lifecycle
- Seccomp profile application

### Integration Tests
- Web scraping in Ornstein
- Code execution in Smough
- Nested Ornstein-in-Smough
- Isolation level selection logic
- Permission system integration

### Security Tests
- SSRF prevention (localhost, private IPs)
- Resource exhaustion resistance
- Host escape attempts (container breakout)
- Syscall filtering effectiveness
- Network allowlist enforcement

## Performance Considerations

| Layer | Overhead | Typical Latency | Use Case |
|-------|----------|-----------------|----------|
| None | 0% | <1ms | Trusted local operations |
| Ornstein | ~5-10% | 10-50ms | Web scraping, API calls |
| Smough | ~50-100% | 500-2000ms | Untrusted code execution |

**Optimization:** Keep Smough containers warm between executions (container pooling)

## Comparison to Existing Systems

| System | Approach | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| **firejail** | Namespace isolation | Lightweight, no root | Limited to Linux |
| **bubblewrap** | Seccomp + namespaces | Sandbox without root | Linux-only |
| **Docker** | Full containerization | Strong isolation, portable | Heavy overhead |
| **gVisor** | User-space kernel | VM-level isolation | Complex, slower |
| **Ornstein** | Process + resource limits | Fast, cross-platform | Medium isolation |
| **Smough** | Docker + seccomp | Strong isolation, audit | High overhead |

## Future Enhancements

1. **Container pooling**: Warm containers reduce Smough latency
2. **eBPF monitoring**: Runtime behavior analysis in containers
3. **VM-level isolation**: Firecracker/gVisor for maximum security
4. **Trust scoring**: Automatic isolation level selection based on action risk
5. **Isolation escape detection**: Monitor for breakout attempts

## References

- Dark Souls boss mechanics: Ornstein (agile) + Smough (heavy)
- Docker security best practices
- Seccomp profiles for container hardening
- Linux namespaces and cgroups
- SSRF prevention techniques

---

**Next Steps:** Implement Phase 1 (Ornstein Layer) with process isolation and resource limiting.
