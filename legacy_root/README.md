# Legacy root files and relocation status

The active application/package source is under `src/autobliteratus/`. The repository also contains root-level `.py` files from earlier prototype iterations. They are outside setuptools' configured `src` package discovery and outside the runtime toggle registry scan.

These root files have **not** been moved by this change. Moving a large set safely requires complete source-preserving Git renames plus reference and test updates; copies or placeholder files are not equivalent to moves. Until that work is performed with a Git-aware move operation, leave these files intact and treat them as historical/prototype material, not active package code.

For a relocation, inventory each file, compare it with its corresponding package module, inspect imports and references, preserve unique implementation/history under `legacy_root/`, then remove the old path in the same commit only after validating the result. Do not archive duplicates blindly and do not infer irrelevance from filename typos.
