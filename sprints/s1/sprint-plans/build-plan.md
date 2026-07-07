Finalized - DO NOT EDIT

# Build Plan

1. Generate the custom "Animus License" by appending an "Additional Terms" section (as permitted by GPLv3 Section 7) to the GPLv3 text, explicitly integrating Apache 2.0's trademark and patent protections. Replace the Dual License in `LICENSE` with this new custom license.
2. Create `ecosystem.json` in the root of the Animus repo. This will list out the repositories in the Animus ecosystem (e.g. `crussella0129/Animus-Model-Lab`, `crussella0129/Banquo`).
3. Create a PowerShell script `scripts/sync-license.ps1` that:
   - Reads `ecosystem.json`.
   - Clones or updates each repository in a temporary directory.
   - Copies the master `LICENSE` file into each repository, overwriting any existing one.
   - Commits and pushes the changes (if any) to the respective repositories.
