# KV Caching

Research scaffold for separating readout distortion from trajectory distortion
in KV-cache compression for reasoning models.

Local machines are for source, configuration, documentation, linting, and
CPU-only unit tests. Models, datasets, caches, and experiments are remote-only.
Read [remoteserver.md](remoteserver.md) before setup, synchronization, or launch.

## Local checks

```bash
make lint
make test
make dryrun
```

## Remote workflow

The selected workflow is manual section 11 Policy B (rsync-first): edit and
test locally, preview and apply a source-only rsync, run remotely, then commit
stable changes locally. See [docs/WORKFLOW.md](docs/WORKFLOW.md).

Scientific claims and analysis choices are frozen in
[PREREGISTRATION.md](PREREGISTRATION.md). Heavy commands require explicit
`--i-understand` confirmation. On the shared remote, launchers also require a
verified `KVCACHE_PERSONAL_FREE_GB`; filesystem-wide `df` space is insufficient.
