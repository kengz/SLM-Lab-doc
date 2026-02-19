# Running Benchmarks 📋

This guide covers how to run reproducible benchmarks with SLM Lab, including hyperparameter search methodology and best practices.

## Quick Start

After [installation](../setup/installation.md), copy `SPEC_FILE` and `SPEC_NAME` from result tables in the [benchmark pages](../benchmark-results/public-benchmark-data.md).

### Running Benchmarks

**Local** - runs on your machine (Classic Control: minutes):
```bash
slm-lab run SPEC_FILE SPEC_NAME train
```

**Remote** - cloud GPU via [dstack](https://dstack.ai), auto-syncs to HuggingFace:
```bash
source .env && slm-lab run-remote --gpu SPEC_FILE SPEC_NAME train -n NAME
```

Remote setup: `cp .env.example .env` then set `HF_TOKEN`. See [Remote Training](remote-training.md) for dstack config.

{% hint style="info" %}
**Recommended:** Use `run-remote` for MuJoCo and Atari benchmarks. Cloud GPUs are faster and cheaper than local training for longer runs.
{% endhint %}

### Atari

PPO, SAC, and A2C all support Atari. Each algorithm has a template spec file. Use `-s env=ENV` to substitute:

```bash
source .env && slm-lab run-remote --gpu -s env=ALE/Pong-v5 slm_lab/spec/benchmark_arc/ppo/ppo_atari_arc.yaml ppo_atari_arc train -n pong
```

### Download Results

Trained models and metrics sync to [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark). Pull locally:
```bash
source .env && slm-lab pull SPEC_NAME
slm-lab list  # see available experiments
```

### Replay Trained Model

```bash
slm-lab run slm_lab/spec/benchmark_arc/ppo/ppo_cartpole_arc.yaml ppo_cartpole_arc enjoy@data/ppo_cartpole_arc_*/ppo_cartpole_arc_t0_spec.yaml
```

{% hint style="info" %}
**Authoritative source:** [BENCHMARKS.md](https://github.com/kengz/SLM-Lab/blob/master/docs/BENCHMARKS.md) in the repository contains exact reproduction commands, current results, and HuggingFace links.
{% endhint %}

## Standardized Settings

Fair comparison requires consistent configurations across environment categories:

| Category | `num_envs` | `max_frame` | `log_frequency` | `max_session` |
|----------|------------|-------------|-----------------|---------------|
| Classic Control | 4 | 2e5-3e5 | 500 | 4 |
| Box2D | 8 | 3e5 | 1000 | 4 |
| MuJoCo | 16 | 1e6-10e6 | 10000 | 4 |
| Atari | 16 | 10e6 | 10000 | 4 |

{% hint style="warning" %}
**Before running:** Verify spec settings match the table above. Inconsistent settings make results incomparable.
{% endhint %}

## Three-Stage Search Process

When tuning hyperparameters or adding new environments, use this systematic approach:

| Stage | Mode | Config | Purpose |
|-------|------|--------|---------|
| **1. ASHA** | `search` | `max_session=1`, `search_scheduler` enabled | Wide exploration with early termination |
| **2. Multi** | `search` | `max_session=4`, NO `search_scheduler` | Validate top configs with multiple seeds |
| **3. Final** | `train` | Best hyperparameters committed to spec | Confirmation run for benchmark table |

### Stage 1: ASHA Search

ASHA (Asynchronous Successive Halving) terminates unpromising trials early, focusing compute on promising configurations.

```yaml
meta:
  max_session: 1
  max_trial: 16
  search_resources:
    cpu: 2
    gpu: 0.25
  search_scheduler:
    grace_period: 500000
    reduction_factor: 3
search:
  agent.algorithm.lam__uniform: [0.7, 0.98]
  agent.algorithm.entropy_coef_spec.start_val__loguniform: [0.005, 0.03]
  agent.net.optim_spec.lr__loguniform: [1e-4, 5e-4]
```

```bash
slm-lab run spec.yaml spec_name search                                        # local
source .env && slm-lab run-remote --gpu spec.yaml spec_name search -n NAME    # remote
```

### Stage 2: Multi-Seed Validation

After ASHA, validate top 3-5 configurations with multiple seeds (no early stopping):

```yaml
meta:
  max_session: 4
  max_trial: 5
```

Single runs can be lucky—averaging 4 independent runs reveals true performance.

### Stage 3: Final Validation

Update spec defaults with best hyperparameters, then run in train mode:

```bash
slm-lab run spec.yaml spec_name train                                        # local
source .env && slm-lab run-remote --gpu spec.yaml spec_name train -n NAME    # remote
```

{% hint style="warning" %}
**Never use raw search results in benchmark tables.** Always run a final validation with committed spec file.
{% endhint %}

## Search Space Sizing

**Rule: ~3-4 trials per search dimension minimum.**

| `max_trial` | Max Dimensions | Use Case |
|-------------|----------------|----------|
| 8 | 2-3 | Focused refinement |
| 12-16 | 3-4 | Typical search |
| 20 | 5 | Wide exploration |
| 30 | 6-7 | Broad ASHA search |

### High-Impact Hyperparameters

Focus on these first—they have the largest effect on performance:

| Priority | Parameter | Path | Typical Range |
|----------|-----------|------|---------------|
| **1** | Learning rate | `agent.net.optim_spec.lr` | 1e-5 to 1e-3 |
| **2** | Discount factor | `agent.algorithm.gamma` | 0.98-0.999 |
| **3** | GAE lambda | `agent.algorithm.lam` | 0.9-0.99 |
| 4 | Entropy coefficient | `agent.algorithm.entropy_coef_spec.start_val` | 0.001-0.1 |
| 5 | Clip epsilon | `agent.algorithm.clip_eps_spec.start_val` | 0.1-0.3 |

**Less impactful** (fix based on successful runs): `minibatch_size`, `training_epoch`, network architecture.

{% hint style="info" %}
**Iterative narrowing:** After finding good ranges, narrow the search space and re-run rather than continuing broad exploration.
{% endhint %}

## Grace Period by Environment

The `grace_period` determines minimum frames before ASHA can terminate trials:

| Environment | `grace_period` | Reasoning |
|-------------|----------------|-----------|
| Classic Control | 10000-50000 | Fast learning, quick signal |
| Box2D | 50000-100000 | Medium complexity |
| MuJoCo | 100000-1000000 | Slower learning curves |
| Atari | 500000-1000000 | Need significant training for signal |

## Template Specs

Template specs use `${var}` placeholders for flexibility across similar environments:

```bash
# MuJoCo template
slm-lab run -s env=HalfCheetah-v5 -s max_frame=10e6 slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml ppo_mujoco_arc train

# Atari template
slm-lab run -s env=ALE/Qbert-v5 slm_lab/spec/benchmark_arc/ppo/ppo_atari_arc.yaml ppo_atari_arc train
```

| Template | Variables | Environments |
|----------|-----------|--------------|
| `ppo_mujoco_arc.yaml` | `env`, `max_frame` | All 11 MuJoCo |
| `ppo_atari_arc.yaml` | `env` | All 57 Atari games |
| `sac_atari_arc.yaml` | `env` | All Atari games |
| `a2c_atari_arc.yaml` | `env` | All Atari games |

## MuJoCo Tips

### Unified vs Individual Specs

- **`ppo_mujoco`**: HalfCheetah, Walker2d, Humanoid, HumanoidStandup (gamma=0.99, lam=0.95)
- **`ppo_mujoco_longhorizon`**: Reacher, Pusher (gamma=0.997, lam=0.97)
- **Individual specs**: Hopper, Swimmer, Ant—each has environment-specific tuning

### Common Issues

| Problem | Solution |
|---------|----------|
| Reward not improving | Try higher `training_iter` (8-16) for more gradient updates |
| Unstable learning | Try lower learning rate or enable `clip_vloss: true` |
| Large reward variance | Enable `normalize_v_targets: true` for value normalization |

## Atari Tips

### Lambda Variants

Different games benefit from different lambda values:

| Spec Name | Lambda | Best For |
|-----------|--------|----------|
| `ppo_atari_arc` | 0.95 | Strategic games (Qbert, Seaquest) |
| `ppo_atari_lam85_arc` | 0.85 | Mixed games (MsPacman) |
| `ppo_atari_lam70_arc` | 0.70 | Action games (Breakout, Pong) |

**Best practice:** Test all three variants per game; use the best result.

### v5 Environment Difficulty

Gymnasium ALE v5 uses sticky actions (25% repeat probability) per Machado et al. 2018. This makes environments harder than OpenAI Gym v4—expect 10-40% lower scores.

## Troubleshooting

### When Progress Stalls

1. **Check GPU metrics** (`dstack metrics <run-name>`)—low GPU util means bottleneck in env stepping or config issue
2. **Compare with successful specs**—review what worked for similar environments
3. **Look for patterns**—same failure across runs suggests framework issue, not hyperparameters
4. **Research reference implementations**—check [CleanRL](https://github.com/vwxyzjn/cleanrl) or [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) configs
5. **Kill unpromising runs early**—iterate faster with new approaches

### Common Mistakes

| Mistake | Fix |
|---------|-----|
| Too many search dimensions | Focus on 2-3 high-impact parameters per search |
| Skipping multi-seed validation | Always run `max_session=4` before finalizing |
| Using search results directly | Always run final `train` mode with committed spec |
| Inconsistent settings | Verify spec matches standardized settings table |

## Recording Results

After a successful run:

1. **Extract final score** from logs:
   ```bash
   dstack logs my-experiment | grep "trial_metrics"
   # Output: trial_metrics: frame:1.00e+07 | total_reward_ma:15094 | ...
   ```

2. **Pull results**:
   ```bash
   slm-lab pull spec_name
   ```

3. **Update spec defaults** with best hyperparameters

4. **Commit spec file** for reproducibility

## Algorithms

| Algorithm | Type | Best For | Validated Environments |
|-----------|------|----------|------------------------|
| **REINFORCE** | On-policy | Learning/teaching | Classic |
| **SARSA** | On-policy | Tabular-like | Classic |
| **DQN/DDQN+PER** | Off-policy | Discrete actions | Classic, Box2D, Atari |
| **A2C** | On-policy | Fast iteration | Classic, Box2D, Atari (57) |
| **PPO** | On-policy | General purpose | Classic, Box2D, MuJoCo (11), Atari (57) |
| **SAC** | Off-policy | Continuous + discrete | Classic, Box2D, MuJoCo, Atari (48) |

## Environments

| Category | Examples | Difficulty | Docs |
|----------|----------|------------|------|
| **Classic Control** | CartPole, Pendulum, Acrobot | Easy | [Gymnasium Classic](https://gymnasium.farama.org/environments/classic_control/) |
| **Box2D** | LunarLander, BipedalWalker | Medium | [Gymnasium Box2D](https://gymnasium.farama.org/environments/box2d/) |
| **MuJoCo** | Hopper, HalfCheetah, Humanoid | Hard | [Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/) |
| **Atari** | Qbert, MsPacman, and 57 more | Varied | [ALE](https://ale.farama.org/environments/) |

## Benchmark Spec Reference

All benchmark specs are in [slm_lab/spec/benchmark_arc/](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark_arc), organized by algorithm.

### REINFORCE / SARSA

Simple algorithms for learning fundamentals. CartPole only.

| Algorithm | Spec File | Spec Names |
|-----------|-----------|------------|
| REINFORCE | [reinforce_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/reinforce/reinforce_arc.yaml) | `reinforce_cartpole_arc` |
| SARSA | [sarsa_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sarsa/sarsa_arc.yaml) | `sarsa_epsilon_greedy_cartpole_arc`, `sarsa_boltzmann_cartpole_arc` |

### DQN Family

Value-based algorithms for discrete action spaces.

| Category | Spec File | Spec Names |
|----------|-----------|------------|
| Classic | [dqn_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml) | `dqn_boltzmann_cartpole_arc`, `dqn_epsilon_greedy_acrobot_arc`, `ddqn_per_acrobot_arc`, etc. |
| Box2D | [dqn_box2d_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml) | `dqn_concat_lunar_arc`, `ddqn_per_concat_lunar_arc` |

### A2C

On-policy actor-critic with synchronized updates using GAE (Generalized Advantage Estimation).

| Category | Spec File | Spec Names |
|----------|-----------|------------|
| Classic | [a2c_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml) | `a2c_gae_cartpole_arc`, `a2c_gae_acrobot_arc`, `a2c_gae_pendulum_arc`, `a2c_gae_lunar_arc` |
| Atari | [a2c_atari_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/a2c/a2c_atari_arc.yaml) | `a2c_gae_atari_arc` (template) |

### PPO

Proximal Policy Optimization -- robust across all environment types.

| Category | Spec File | Spec Names |
|----------|-----------|------------|
| Classic | [ppo_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml) | `ppo_cartpole_arc`, `ppo_acrobot_arc`, `ppo_pendulum_arc` |
| Box2D | [ppo_box2d_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_box2d_arc.yaml) | `ppo_lunar_arc`, `ppo_bipedalwalker_arc` |
| MuJoCo | [ppo_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | `ppo_mujoco_arc` (template), `ppo_hopper_arc`, `ppo_ant_arc`, etc. |
| Atari | [ppo_atari_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_atari_arc.yaml) | `ppo_atari_arc` (template), `ppo_atari_lam85_arc`, `ppo_atari_lam70_arc` |

### SAC

Soft Actor-Critic -- works for both continuous and discrete action spaces.

| Category | Spec File | Spec Names |
|----------|-----------|------------|
| Classic | [sac_classic_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml) | `sac_cartpole_arc`, `sac_acrobot_arc`, `sac_pendulum_arc` |
| Box2D | [sac_box2d_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_box2d_arc.yaml) | `sac_lunar_arc`, `sac_bipedalwalker_arc` |
| MuJoCo | [sac_mujoco_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | `sac_mujoco_arc` (template), `sac_halfcheetah_arc`, `sac_hopper_arc`, etc. |
| Atari | [sac_atari_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_atari_arc.yaml) | `sac_atari_arc` (template) |

## Performance Results

For scores, training curves, and trained models:

- [Discrete Benchmark](../benchmark-results/discrete-benchmark.md) — Classic Control, Box2D
- [Continuous Benchmark](../benchmark-results/continuous-benchmark.md) — MuJoCo
- [Atari Benchmark](../benchmark-results/atari-benchmark.md) — 57 Atari games
- [Public Benchmark Data](../benchmark-results/public-benchmark-data.md) — HuggingFace download links
