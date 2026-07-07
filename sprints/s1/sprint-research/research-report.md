# Research Report: Sprint 1 - Custom License vs Dual License

## Goals
1. Draft a "best of both worlds" custom license.
2. Establish a sync mechanism for other repositories in the Animus ecosystem.

## Custom License Exploration
To get the "best of both worlds" (GPL 3.0 copyleft + Apache 2.0 patent/trademark protections), we can create a custom license document.
- **Base**: GNU General Public License v3.0 (GPLv3).
- **Addition**: Section 7 of GPLv3 explicitly allows appending "Additional Terms" to the license. We can append an "Animus Exception" or "Animus Additional Terms" that explicitly enforces Apache 2.0's exact patent grant and trademark clauses.
- **Why this works legally**: GPLv3 was specifically designed to be compatible with Apache 2.0. By taking GPLv3 and using its built-in Section 7 to add the explicit Trademark clause (which GPL doesn't cover natively) and ensuring the Patent Grant explicitly mirrors Apache's wording, you get an airtight custom license.

## Pull Mechanism Exploration
Once the license text is finalized, we need a way to propagate it.
- A PowerShell script (`sync-license.ps1`) that clones repositories from a defined list (`ecosystem.json`), copies the master `LICENSE` file, and pushes the changes, seems to be the most robust way to ensure physical files are present in all repositories (which platforms like GitHub expect in order to parse the repository's license).
