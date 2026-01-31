# Contributing

Thank you for your interest in contributing to SLM Lab!

## Contribution Tracks

### 1. Run Benchmark Experiments

Help validate algorithms across environments. The easiest way to start:

```bash
# Pick an environment from the benchmark pages
slm-lab run slm_lab/spec/benchmark/ppo/ppo_hopper.json ppo_hopper train

# Upload results
source .env
slm-lab push data/ppo_hopper_*
```

See [Benchmark Results](../benchmark-results/public-benchmark-data.md) for what's needed.

### 2. Reproduce Published Results

Reproducibility is crucial. Pick a result from our benchmarks and verify it:

1. Download the spec: `slm-lab pull ppo_hopper`
2. Run: `slm-lab run _ _ train@data/ppo_hopper_*/ppo_hopper_spec.json`
3. Compare your results to published scores
4. Report discrepancies as [issues](https://github.com/kengz/SLM-Lab/issues)

### 3. Implement Features

Check [GitHub Issues](https://github.com/kengz/SLM-Lab/issues) for feature requests. Good first issues:

- Add new environment wrappers
- Implement algorithm variants
- Improve documentation
- Add unit tests

### 4. Fix Bugs

Found a bug? Help us fix it:

1. Check if it's already reported in [issues](https://github.com/kengz/SLM-Lab/issues)
2. Create a minimal reproduction case
3. Submit a PR with the fix and a test

## Development Workflow

### Setup

```bash
git clone https://github.com/kengz/SLM-Lab.git
cd SLM-Lab
uv sync
```

### Making Changes

```bash
# Create a branch
git checkout -b feature/your-feature

# Make changes, run tests
uv run pytest

# Format code
uv run ruff format .
uv run ruff check . --fix

# Commit with conventional format
git commit -m "feat: add new feature"
```

### Pull Request Guidelines

1. **Small, focused PRs** - One feature or fix per PR
2. **Tests required** - Add tests for new functionality
3. **Documentation** - Update docs if behavior changes
4. **Pass CI** - All tests must pass

## Design Principles

SLM Lab follows these principles:

| Principle | Meaning |
|-----------|---------|
| **Modularity** | Components are reusable and composable |
| **Simplicity** | Code matches how algorithms are described in papers |
| **Analytical clarity** | Results should be easy to understand and compare |
| **Reproducibility** | Spec + git SHA = exact reproduction |

When contributing, ask: "Does this make SLM Lab simpler and more modular?"

## Using AI Coding Assistants

SLM Lab supports development with AI assistants like Claude Code. The repository includes:

- **`CLAUDE.md`** - Project context and agent instructions
- **`.claude/skills/`** - Specialized skills for benchmark work

Agents can help with:
- Running benchmarks and updating results
- Implementing features following codebase patterns
- Writing tests and documentation

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/kengz/SLM-Lab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kengz/SLM-Lab/discussions)
- **Chat**: [Gitter](https://gitter.im/SLM-Lab/SLM-Lab)

## Code of Conduct

We follow the [Contributor Covenant](code-of-conduct.md). Be respectful and constructive.
