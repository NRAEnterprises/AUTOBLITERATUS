# Root prototype archive

This directory is the designated archive path for the repository's legacy root-level prototype modules.

The current runtime and package discovery are anchored under `src/autobliteratus/`, and the active registry scans only `src/autobliteratus/tuning/`. The root-level Python modules are therefore retained as historical and migration-reference material rather than live runtime code.

This archive is intentionally a staging area. The GitHub file API does not support a true Git rename/delete of the original root files inside this environment, so the safe repository state for now is:

- canonical active code remains under `src/autobliteratus/`
- the original root-level prototype files are preserved in place until a local git checkout is used for a real `git mv`/archive cleanup
- this directory records the archive boundary and the reason for the staged move

This archive is not a runtime import path and must not be treated as active package code.
