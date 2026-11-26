# CI verification — 2026-11-25

This record covers successful checks only for `main` at `6a5b924ec9fba50e9edd50ebf71f7ceea737821e`. It does not make claims about external deployment checks outside this report's scope.

## Successful GitHub Actions runs

- [Dependency Graph](https://github.com/NRAEnterprises/AUTOBLITERATUS/actions/runs/45709353974) — success; `update-pip-graph` completed successfully.
- [CodeQL](https://github.com/NRAEnterprises/AUTOBLITERATUS/actions/runs/45709345615) — success; C/C++, Python, and JavaScript/TypeScript analysis jobs completed successfully.
- [CodeQL — Code Quality](https://github.com/NRAEnterprises/AUTOBLITERATUS/actions/runs/45709344750) — success; Python and JavaScript/TypeScript analysis jobs completed successfully.
- [Pages build and deployment](https://github.com/NRAEnterprises/AUTOBLITERATUS/actions/runs/45709343991) — success; `build`, `deploy`, and `report-build-status` jobs completed successfully.

All nine successful check runs associated with these workflows reported `completed / success` at verification time.

## Successful local frontend build

In `ui/model-mapping-observatory/`, a fresh locked install and production build completed successfully:

```sh
npm ci --no-audit --no-fund && npm run build
```

Environment: Node `v22.23.4`, npm `10.9.9`. TypeScript compilation and Vite production bundling passed. Vite emitted a non-fatal advisory that the minified JavaScript chunk is approximately 798 kB, above its 500 kB warning threshold.

This verification record does not include or alter the pending Cloudflare follow-up note.
