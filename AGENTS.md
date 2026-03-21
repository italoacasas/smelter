# AGENTS.md

## Repo Intent

Smelter is a GPU inference service built around SGLang, Docker Compose, and a local Hugging Face model cache. It is intentionally simple — one host, one container, one model at a time.

## Configuration System

Configuration is split across two JSON files and one selector:

- `models.json` — model definitions and shared settings
- `hardware.json` — hardware profiles with per-model tuning
- `.active` — two-line file selecting the active model and hardware profile

`scripts/load-config.sh` reads all three and exports env vars. All scripts source it. `docker-compose.yml` consumes the env vars — no values are hardcoded there.

When adding a model, update `models.json` and add per-model tuning to every hardware profile in `hardware.json`. When adding hardware, add a complete profile to `hardware.json` with `models` entries for each supported model.

## External References

- [SGLang docs](https://docs.sglang.io/)
- [SGLang server arguments](https://docs.sglang.io/advanced_features/server_arguments.html)
- [SGLang cookbook](https://lmsysorg.mintlify.app/cookbook)

## Documentation Rule

Treat `docs/architecture.md` as a living current-state record. Update it when changes affect runtime behavior, API surface, configuration, storage layout, or operational workflows.

## Expectations For Agents

- Do not leave architecture-impacting changes undocumented
- Update `docs/architecture.md` in the same change as the implementation
- Keep it focused on what is implemented now, not future design
- Remove stale statements instead of appending conflicting notes
