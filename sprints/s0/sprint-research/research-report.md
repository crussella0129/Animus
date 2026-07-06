# Research Report: Sprint 0 - Master Repo & Licensing

## Goals
1. Configure `c:\Users\charl\Animus` as the master repository for the Animus Ecosystem.
2. Formulate the "Animus License," combining the best properties of GPL 3.0 and Apache 2.0.
3. Establish a mechanism for all projects in the ecosystem (including `Animus-Model-Lab`) to "pull" from this license.

## Findings
1. **Current State**: The `Animus` repository is currently nearly empty. It contains a standard `LICENSE` (likely Apache 2.0 based on length) and a brief `README.md`.
2. **License Combination**: 
   - **GPL 3.0** provides strong copyleft protections (anyone modifying the software must release their modifications under the same license).
   - **Apache 2.0** provides explicit patent grants and trademark protections but is permissive (not copyleft).
   - The standard legal way to combine these without drafting a legally ambiguous custom license is a **Dual License** (e.g., "Licensed under Apache 2.0 OR GPL 3.0"). Alternatively, a custom "Animus License" can be drafted that takes the GPL 3.0 base and adds the specific patent/trademark clauses of Apache 2.0.
3. **Pulling the License**:
   - Option A: Host the canonical `ANIMUS-LICENSE.txt` here and have other repos link to it in their `README` or a stub `LICENSE` file.
   - Option B: Use Git Submodules. Make `Animus` a submodule in all other projects just for the license (overkill).
   - Option C: Write a GitHub Action or a sync script in this master repo that pushes license updates to all ecosystem repos.
   - Option D: In each sub-project, the `LICENSE` file simply contains: "This software is licensed under the Animus License. See https://github.com/crussella0129/Animus/blob/main/LICENSE for details."

## Open Questions
- Do we want a true Dual License (User's choice) or a Custom Hybrid License (Copyleft + Patent Grants)?
- What is the preferred technical mechanism for "pulling" the license? (Linking vs Scripted Syncing)
