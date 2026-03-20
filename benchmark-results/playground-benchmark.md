# MuJoCo Playground Benchmark 🎮

## MuJoCo Playground PPO Benchmark Results

SLM Lab v5.3 validates PPO on [MuJoCo Playground](https://google-deepmind.github.io/mujoco_playground/) — Google DeepMind's GPU-accelerated simulation platform. MuJoCo Playground uses the MJWarp backend (Warp-accelerated MJX) for physics, enabling massively parallel training with 2048 environments on GPU.

SLM Lab wraps Playground environments as `gymnasium.VectorEnv` with DLPack zero-copy JAX→PyTorch transfer. All 54 environments use the `playground/` prefix in specs.

Results below are from March 2026 benchmark runs. All trained models and metrics are publicly available on [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark).

### Methodology

Results show **Trial-level** performance:

1. **Trial** = 4 Sessions with different random seeds
2. **Session** = One complete training run
3. **Score** = Final 100-checkpoint moving average (`total_reward_ma`)

The trial score is the mean across 4 sessions.

### Standardized Settings

| Category | num_envs | max_frame | log_frequency |
|----------|----------|-----------|---------------|
| Playground | 2048 | 100e6 | 10000 |

### Spec File

**Spec file**: [ppo_playground.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_playground.yaml) — all envs via `-s env=playground/ENV`

### Running Benchmarks

```bash
# Requires: uv sync --group playground (JAX + MuJoCo Playground + MJWarp)

source .env && slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/ppo/ppo_playground.yaml SPEC_NAME train \
  -s env=playground/ENV -s max_frame=100000000 -n NAME
```

### Installation

```bash
uv sync --group playground
```

This adds JAX, MuJoCo Playground, and MJWarp dependencies. Requires a CUDA GPU.

---

## Phase 5.1: DM Control Suite (25 envs)

Classic control and locomotion tasks from the DeepMind Control Suite, ported to MJWarp GPU simulation.

| ENV | MA | SPEC_NAME | HF Data |
|-----|-----|-----------|---------|
| playground/AcrobotSwingup | 253.24 | ppo_playground_vnorm | [ppo_playground_acrobotswingup_2026_03_12_175809](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_acrobotswingup_2026_03_12_175809) |
| playground/AcrobotSwingupSparse | 146.98 | ppo_playground_vnorm | [ppo_playground_vnorm_acrobotswingupsparse_2026_03_14_161212](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_vnorm_acrobotswingupsparse_2026_03_14_161212) |
| playground/BallInCup | 942.44 | ppo_playground_vnorm | [ppo_playground_ballincup_2026_03_12_105443](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_ballincup_2026_03_12_105443) |
| playground/CartpoleBalance | 968.23 | ppo_playground_vnorm | [ppo_playground_cartpolebalance_2026_03_12_141924](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_cartpolebalance_2026_03_12_141924) |
| playground/CartpoleBalanceSparse | 995.34 | ppo_playground_constlr | [ppo_playground_constlr_cartpolebalancesparse_2026_03_14_000352](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_constlr_cartpolebalancesparse_2026_03_14_000352) |
| playground/CartpoleSwingup | 729.09 | ppo_playground_constlr | [ppo_playground_constlr_cartpoleswingup_2026_03_17_041102](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_constlr_cartpoleswingup_2026_03_17_041102) |
| playground/CartpoleSwingupSparse | 521.98 | ppo_playground_constlr | [ppo_playground_constlr_cartpoleswingupsparse_2026_03_13_233449](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_constlr_cartpoleswingupsparse_2026_03_13_233449) |
| playground/CheetahRun | 883.44 | ppo_playground_vnorm | [ppo_playground_vnorm_cheetahrun_2026_03_14_161211](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_vnorm_cheetahrun_2026_03_14_161211) |
| playground/FingerSpin | 713.35 | ppo_playground_fingerspin | [ppo_playground_fingerspin_fingerspin_2026_03_13_033911](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_fingerspin_fingerspin_2026_03_13_033911) |
| playground/FingerTurnEasy | 663.58 | ppo_playground_vnorm | [ppo_playground_fingerturneasy_2026_03_12_175835](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_fingerturneasy_2026_03_12_175835) |
| playground/FingerTurnHard | 590.43 | ppo_playground_vnorm_constlr | [ppo_playground_vnorm_constlr_fingerturnhard_2026_03_16_234509](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_vnorm_constlr_fingerturnhard_2026_03_16_234509) |
| playground/FishSwim | 580.57 | ppo_playground_vnorm_constlr_clip03 | [ppo_playground_vnorm_constlr_clip03_fishswim_2026_03_14_002112](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_vnorm_constlr_clip03_fishswim_2026_03_14_002112) |
| playground/HopperHop | 22.00 | ppo_playground_vnorm | [ppo_playground_hopperhop_2026_03_12_110855](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_hopperhop_2026_03_12_110855) |
| playground/HopperStand | 237.15 | ppo_playground_vnorm | [ppo_playground_vnorm_hopperstand_2026_03_14_095438](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_vnorm_hopperstand_2026_03_14_095438) |
| playground/HumanoidRun | 18.83 | ppo_playground_humanoid | [ppo_playground_humanoid_humanoidrun_2026_03_14_115522](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_humanoid_humanoidrun_2026_03_14_115522) |
| playground/HumanoidStand | 114.86 | ppo_playground_humanoid | [ppo_playground_humanoid_humanoidstand_2026_03_14_115516](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_humanoid_humanoidstand_2026_03_14_115516) |
| playground/HumanoidWalk | 47.01 | ppo_playground_humanoid | [ppo_playground_humanoid_humanoidwalk_2026_03_14_172235](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_humanoid_humanoidwalk_2026_03_14_172235) |
| playground/PendulumSwingup | 637.46 | ppo_playground_pendulum | [ppo_playground_pendulum_pendulumswingup_2026_03_13_033818](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_pendulum_pendulumswingup_2026_03_13_033818) |
| playground/PointMass | 868.09 | ppo_playground_vnorm_constlr | [ppo_playground_vnorm_constlr_pointmass_2026_03_14_095452](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_vnorm_constlr_pointmass_2026_03_14_095452) |
| playground/ReacherEasy | 955.08 | ppo_playground_vnorm | [ppo_playground_reachereasy_2026_03_12_122115](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_reachereasy_2026_03_12_122115) |
| playground/ReacherHard | 946.99 | ppo_playground_vnorm | [ppo_playground_reacherhard_2026_03_12_123226](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_reacherhard_2026_03_12_123226) |
| playground/SwimmerSwimmer6 | 591.13 | ppo_playground_vnorm_constlr | [ppo_playground_vnorm_constlr_swimmerswimmer6_2026_03_14_000406](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_vnorm_constlr_swimmerswimmer6_2026_03_14_000406) |
| playground/WalkerRun | 759.71 | ppo_playground_vnorm | [ppo_playground_vnorm_walkerrun_2026_03_14_161354](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_vnorm_walkerrun_2026_03_14_161354) |
| playground/WalkerStand | 948.35 | ppo_playground_vnorm | [ppo_playground_vnorm_walkerstand_2026_03_14_161415](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_vnorm_walkerstand_2026_03_14_161415) |
| playground/WalkerWalk | 945.31 | ppo_playground_vnorm | [ppo_playground_vnorm_walkerwalk_2026_03_14_161338](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_vnorm_walkerwalk_2026_03_14_161338) |

| | | |
|---|---|---|
| ![AcrobotSwingup](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/AcrobotSwingup_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![AcrobotSwingupSparse](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/AcrobotSwingupSparse_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![BallInCup](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/BallInCup_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![CartpoleBalance](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/CartpoleBalance_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![CartpoleBalanceSparse](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/CartpoleBalanceSparse_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![CartpoleSwingup](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/CartpoleSwingup_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![CartpoleSwingupSparse](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/CartpoleSwingupSparse_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![CheetahRun](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/CheetahRun_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![FingerSpin](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/FingerSpin_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![FingerTurnEasy](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/FingerTurnEasy_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![FingerTurnHard](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/FingerTurnHard_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![FishSwim](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/FishSwim_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![HopperHop](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/HopperHop_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![HopperStand](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/HopperStand_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![HumanoidRun](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/HumanoidRun_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![HumanoidStand](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/HumanoidStand_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![HumanoidWalk](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/HumanoidWalk_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![PendulumSwingup](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/PendulumSwingup_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![PointMass](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/PointMass_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![ReacherEasy](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/ReacherEasy_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![ReacherHard](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/ReacherHard_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![SwimmerSwimmer6](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/SwimmerSwimmer6_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![WalkerRun](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/WalkerRun_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![WalkerStand](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/WalkerStand_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![WalkerWalk](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/WalkerWalk_multi_trial_graph_mean_returns_ma_vs_frames.png) | | |

---

## Phase 5.2: Locomotion Robots (19 envs)

Real-world robot locomotion — quadrupeds (Go1, Spot, Barkour) and humanoids (H1, G1, T1, Op3, Apollo, BerkeleyHumanoid) on flat and rough terrain.

| ENV | MA | SPEC_NAME | HF Data |
|-----|-----|-----------|---------|
| playground/ApolloJoystickFlatTerrain | 17.44 | ppo_playground_loco_precise | [ppo_playground_loco_precise_apollojoystickflatterrain_2026_03_14_210939](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_precise_apollojoystickflatterrain_2026_03_14_210939) |
| playground/BarkourJoystick | 0.0 | ppo_playground_loco | [ppo_playground_loco_barkourjoystick_2026_03_14_194525](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_barkourjoystick_2026_03_14_194525) |
| playground/BerkeleyHumanoidJoystickFlatTerrain | 32.29 | ppo_playground_loco_precise | [ppo_playground_loco_precise_berkeleyhumanoidjoystickflatterrain_2026_03_14_213019](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_precise_berkeleyhumanoidjoystickflatterrain_2026_03_14_213019) |
| playground/BerkeleyHumanoidJoystickRoughTerrain | 21.25 | ppo_playground_loco_precise | [ppo_playground_loco_precise_berkeleyhumanoidjoystickroughterrain_2026_03_15_150211](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_precise_berkeleyhumanoidjoystickroughterrain_2026_03_15_150211) |
| playground/G1JoystickFlatTerrain | 1.85 | ppo_playground_loco_precise | [ppo_playground_loco_precise_g1joystickflatterrain_2026_03_15_150219](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_precise_g1joystickflatterrain_2026_03_15_150219) |
| playground/G1JoystickRoughTerrain | -2.75 | ppo_playground_loco_precise | [ppo_playground_loco_precise_g1joystickroughterrain_2026_03_19_015137](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_precise_g1joystickroughterrain_2026_03_19_015137) |
| playground/Go1Footstand | 23.48 | ppo_playground_loco_precise | [ppo_playground_loco_precise_go1footstand_2026_03_16_174009](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_precise_go1footstand_2026_03_16_174009) |
| playground/Go1Getup | 18.16 | ppo_playground_loco_go1 | [ppo_playground_loco_go1_go1getup_2026_03_16_132801](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_go1_go1getup_2026_03_16_132801) |
| playground/Go1Handstand | 17.88 | ppo_playground_loco_precise | [ppo_playground_loco_precise_go1handstand_2026_03_16_155437](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_precise_go1handstand_2026_03_16_155437) |
| playground/Go1JoystickFlatTerrain | 0.0 | ppo_playground_loco | [ppo_playground_loco_go1joystickflatterrain_2026_03_14_204658](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_go1joystickflatterrain_2026_03_14_204658) |
| playground/Go1JoystickRoughTerrain | 0.00 | ppo_playground_loco | [ppo_playground_loco_go1joystickroughterrain_2026_03_15_150321](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_go1joystickroughterrain_2026_03_15_150321) |
| playground/H1InplaceGaitTracking | 11.95 | ppo_playground_loco_precise | [ppo_playground_loco_precise_h1inplacegaittracking_2026_03_16_170327](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_precise_h1inplacegaittracking_2026_03_16_170327) |
| playground/H1JoystickGaitTracking | 31.11 | ppo_playground_loco_precise | [ppo_playground_loco_precise_h1joystickgaittracking_2026_03_16_170412](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_precise_h1joystickgaittracking_2026_03_16_170412) |
| playground/Op3Joystick | 0.00 | ppo_playground_loco | [ppo_playground_loco_op3joystick_2026_03_15_150120](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_op3joystick_2026_03_15_150120) |
| playground/SpotFlatTerrainJoystick | 48.58 | ppo_playground_loco_precise | [ppo_playground_loco_precise_spotflatterrainjoystick_2026_03_16_180747](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_precise_spotflatterrainjoystick_2026_03_16_180747) |
| playground/SpotGetup | 19.39 | ppo_playground_loco | [ppo_playground_loco_spotgetup_2026_03_14_213703](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_spotgetup_2026_03_14_213703) |
| playground/SpotJoystickGaitTracking | 36.90 | ppo_playground_loco | [ppo_playground_loco_spotjoystickgaittracking_2026_03_19_015106](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_spotjoystickgaittracking_2026_03_19_015106) |
| playground/T1JoystickFlatTerrain | 13.42 | ppo_playground_loco_precise | [ppo_playground_loco_precise_t1joystickflatterrain_2026_03_14_220250](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_precise_t1joystickflatterrain_2026_03_14_220250) |
| playground/T1JoystickRoughTerrain | 2.58 | ppo_playground_loco_precise | [ppo_playground_loco_precise_t1joystickroughterrain_2026_03_15_162332](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_precise_t1joystickroughterrain_2026_03_15_162332) |

| | | |
|---|---|---|
| ![ApolloJoystickFlatTerrain](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/ApolloJoystickFlatTerrain_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![BarkourJoystick](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/BarkourJoystick_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![BerkeleyHumanoidJoystickFlatTerrain](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/BerkeleyHumanoidJoystickFlatTerrain_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![G1JoystickFlatTerrain](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/G1JoystickFlatTerrain_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Go1Footstand](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Go1Footstand_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Go1Handstand](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Go1Handstand_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![H1InplaceGaitTracking](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/H1InplaceGaitTracking_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![H1JoystickGaitTracking](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/H1JoystickGaitTracking_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Op3Joystick](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Op3Joystick_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![SpotFlatTerrainJoystick](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/SpotFlatTerrainJoystick_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![SpotGetup](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/SpotGetup_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![SpotJoystickGaitTracking](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/SpotJoystickGaitTracking_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![BerkeleyHumanoidJoystickRoughTerrain](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/BerkeleyHumanoidJoystickRoughTerrain_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Go1Getup](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Go1Getup_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Go1JoystickFlatTerrain](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Go1JoystickFlatTerrain_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Go1JoystickRoughTerrain](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Go1JoystickRoughTerrain_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![T1JoystickFlatTerrain](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/T1JoystickFlatTerrain_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![T1JoystickRoughTerrain](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/T1JoystickRoughTerrain_multi_trial_graph_mean_returns_ma_vs_frames.png) |

---

## Phase 5.3: Manipulation (10 envs)

Robotic manipulation — Panda arm pick/place, Aloha bimanual, Leap dexterous hand, and AeroCube orientation tasks.

| ENV | MA | SPEC_NAME | HF Data |
|-----|-----|-----------|---------|
| playground/AeroCubeRotateZAxis | -3.09 | ppo_playground_loco | [ppo_playground_loco_aerocuberotatezaxis_2026_03_20_012502](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_aerocuberotatezaxis_2026_03_20_012502) |
| playground/AlohaHandOver | 3.65 | ppo_playground_loco | [ppo_playground_loco_alohahandover_2026_03_15_023712](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_alohahandover_2026_03_15_023712) |
| playground/AlohaSinglePegInsertion | 220.93 | ppo_playground_manip_aloha_peg | [ppo_playground_manip_aloha_peg_alohasinglepeginsertion_2026_03_17_122613](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_manip_aloha_peg_alohasinglepeginsertion_2026_03_17_122613) |
| playground/LeapCubeReorient | 74.68 | ppo_playground_loco | [ppo_playground_loco_leapcubereorient_2026_03_15_150420](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_leapcubereorient_2026_03_15_150420) |
| playground/LeapCubeRotateZAxis | 91.65 | ppo_playground_loco | [ppo_playground_loco_leapcuberotatezaxis_2026_03_15_150334](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_leapcuberotatezaxis_2026_03_15_150334) |
| playground/PandaOpenCabinet | 11081.51 | ppo_playground_loco | [ppo_playground_loco_pandaopencabinet_2026_03_15_150318](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_pandaopencabinet_2026_03_15_150318) |
| playground/PandaPickCube | 4586.13 | ppo_playground_loco | [ppo_playground_loco_pandapickcube_2026_03_15_023744](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_pandapickcube_2026_03_15_023744) |
| playground/PandaPickCubeCartesian | 10.58 | ppo_playground_loco | [ppo_playground_loco_pandapickcubecartesian_2026_03_15_023810](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_pandapickcubecartesian_2026_03_15_023810) |
| playground/PandaPickCubeOrientation | 4281.66 | ppo_playground_loco | [ppo_playground_loco_pandapickcubeorientation_2026_03_19_015108](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_pandapickcubeorientation_2026_03_19_015108) |
| playground/PandaRobotiqPushCube | 1.31 | ppo_playground_loco | [ppo_playground_loco_pandarobotiqpushcube_2026_03_15_042131](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_playground_loco_pandarobotiqpushcube_2026_03_15_042131) |

| | | |
|---|---|---|
| ![AeroCubeRotateZAxis](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/AeroCubeRotateZAxis_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![AlohaHandOver](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/AlohaHandOver_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![AlohaSinglePegInsertion](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/AlohaSinglePegInsertion_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![LeapCubeReorient](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/LeapCubeReorient_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![LeapCubeRotateZAxis](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/LeapCubeRotateZAxis_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![PandaOpenCabinet](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/PandaOpenCabinet_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![PandaPickCube](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/PandaPickCube_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![PandaPickCubeCartesian](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/PandaPickCubeCartesian_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![PandaPickCubeOrientation](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/PandaPickCubeOrientation_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![PandaRobotiqPushCube](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/PandaRobotiqPushCube_multi_trial_graph_mean_returns_ma_vs_frames.png) | | |
