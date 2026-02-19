# Atari Environment Benchmark

## A2C, PPO & SAC Atari Results

SLM Lab v5.1 validates A2C, PPO, and SAC on [ALE (Arcade Learning Environment)](https://ale.farama.org/environments/) environments using the TorchArc neural network architecture. The ALE provides 50+ classic Atari 2600 games as standardized RL benchmarks.

| MsPacman | Breakout | Qbert | BeamRider |
|:---:|:---:|:---:|:---:|
| ![MsPacman](https://user-images.githubusercontent.com/8209263/63994685-5cb30d00-caaa-11e9-8f35-78e29a7d60f5.gif) | ![Breakout](https://user-images.githubusercontent.com/8209263/63994695-650b4800-caaa-11e9-9982-2462738caa45.gif) | ![Qbert](https://user-images.githubusercontent.com/8209263/63994672-54f36880-caaa-11e9-9757-7780725b53af.gif) | ![BeamRider](https://user-images.githubusercontent.com/8209263/63994698-689ecf00-caaa-11e9-991f-0a5e9c2f5804.gif) |

**57 games tested** with all results available on [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark).

{% hint style="warning" %}
**v5 vs v4 Difficulty:** Gymnasium ALE v5 is significantly harder than OpenAI Gym's NoFrameskip-v4:
- **Sticky actions** (`repeat_action_probability=0.25`) per [Machado et al. (2018)](https://arxiv.org/abs/1709.06009)
- **Deterministic frame skipping** with proper action handling
- **Stricter termination** conditions

Expect **10-40% lower scores** compared to older benchmarks. Some games (Bowling, Skiing) are much harder in v5.
{% endhint %}

### Methodology

Results show **Trial-level** performance:

1. **Trial** = 4 Sessions with different random seeds
2. **Session** = One complete training run
3. **Score** = Final 100-checkpoint moving average (`total_reward_ma`)

The trial score is the mean across 4 sessions, providing statistically meaningful results.

### Configuration

**Settings**: num_envs 16 | max_session 4 | log_frequency 10000

**Algorithm Specs** (all use Nature CNN [32,64,64] + 512fc):
- **A2C**: [a2c_atari_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/a2c/a2c_atari_arc.yaml) - RMSprop (lr=7e-4), training_frequency=32, max_frame=10e6
- **PPO**: [ppo_atari_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/ppo/ppo_atari_arc.yaml) - AdamW (lr=2.5e-4), minibatch=256, horizon=128, epochs=4, max_frame=10e6
- **SAC**: [sac_atari_arc.yaml](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark_arc/sac/sac_atari_arc.yaml) - Categorical SAC, AdamW (lr=3e-4), training_iter=3, training_frequency=4, max_frame=2e6

{% hint style="info" %}
**SAC uses only 2M frames** vs A2C/PPO's 10M frames. As an off-policy algorithm, SAC is more sample-efficient but slower per frame.
{% endhint %}

**Environment:** Gymnasium ALE v5 with `life_loss_info=true`, sticky actions (`repeat_action_probability=0.25`)

### PPO Lambda Variants

Different games benefit from different lambda values for GAE. All variants use the same spec file:

| SPEC_NAME | Lambda | Best for |
|-----------|--------|----------|
| ppo_atari_arc | 0.95 | Strategic games (default) |
| ppo_atari_lam85_arc | 0.85 | Mixed games |
| ppo_atari_lam70_arc | 0.70 | Action games |

<details>
<summary><b>PPO Lambda Comparison Table</b> - click to expand</summary>

Shows the best PPO lambda variant per game. **Bold** = best score, `-` = not tested.

| ENV | ppo_atari_arc | ppo_atari_lam85_arc | ppo_atari_lam70_arc |
|-----|---------------|---------------------|---------------------|
| ALE/AirRaid-v5 | **7042.84** | - | - |
| ALE/Alien-v5 | **1789.26** | - | - |
| ALE/Amidar-v5 | - | **584.28** | - |
| ALE/Assault-v5 | - | **4448.16** | - |
| ALE/Asterix-v5 | - | **3235.46** | - |
| ALE/Asteroids-v5 | - | **1577.92** | - |
| ALE/Atlantis-v5 | **848087.19** | - | - |
| ALE/BankHeist-v5 | **1058.25** | - | - |
| ALE/BattleZone-v5 | - | **27176.78** | - |
| ALE/BeamRider-v5 | **2761.75** | - | - |
| ALE/Berzerk-v5 | **835.46** | - | - |
| ALE/Bowling-v5 | **45.02** | - | - |
| ALE/Boxing-v5 | **92.18** | - | - |
| ALE/Breakout-v5 | - | - | **326.47** |
| ALE/Carnival-v5 | - | - | **3912.59** |
| ALE/Centipede-v5 | - | - | **4780.75** |
| ALE/ChopperCommand-v5 | **5391.30** | - | - |
| ALE/CrazyClimber-v5 | - | **112094.03** | - |
| ALE/Defender-v5 | - | - | **47894.69** |
| ALE/DemonAttack-v5 | - | - | **19370.38** |
| ALE/DoubleDunk-v5 | **-3.03** | - | - |
| ALE/Enduro-v5 | - | **986.46** | - |
| ALE/FishingDerby-v5 | - | **25.71** | - |
| ALE/Freeway-v5 | **32.42** | - | - |
| ALE/Frostbite-v5 | **284.07** | - | - |
| ALE/Gopher-v5 | - | - | **6500.38** |
| ALE/Gravitar-v5 | **602.58** | - | - |
| ALE/Hero-v5 | - | **22477.89** | - |
| ALE/IceHockey-v5 | **-4.05** | - | - |
| ALE/Jamesbond-v5 | **710.98** | - | - |
| ALE/JourneyEscape-v5 | - | **-1248.98** | - |
| ALE/Kangaroo-v5 | - | - | **10660.35** |
| ALE/Krull-v5 | **7874.33** | - | - |
| ALE/KungFuMaster-v5 | - | - | **28128.04** |
| ALE/MsPacman-v5 | - | **2330.74** | - |
| ALE/NameThisGame-v5 | **6879.23** | - | - |
| ALE/Phoenix-v5 | - | - | **13923.26** |
| ALE/Pong-v5 | - | **16.69** | - |
| ALE/Pooyan-v5 | - | - | **5308.66** |
| ALE/Qbert-v5 | **15460.48** | - | - |
| ALE/Riverraid-v5 | - | **9599.75** | - |
| ALE/RoadRunner-v5 | - | **37980.95** | - |
| ALE/Robotank-v5 | **21.04** | - | - |
| ALE/Seaquest-v5 | **1775.14** | - | - |
| ALE/Skiing-v5 | **-28217.28** | - | - |
| ALE/Solaris-v5 | **2212.78** | - | - |
| ALE/SpaceInvaders-v5 | **892.49** | - | - |
| ALE/StarGunner-v5 | - | - | **49328.73** |
| ALE/Surround-v5 | **-4.47** | - | - |
| ALE/Tennis-v5 | - | **-12.27** | - |
| ALE/TimePilot-v5 | **4432.73** | - | - |
| ALE/Tutankham-v5 | - | **210.87** | - |
| ALE/UpNDown-v5 | - | **147168.80** | - |
| ALE/VideoPinball-v5 | - | - | **38370.30** |
| ALE/WizardOfWor-v5 | **6100.42** | - | - |
| ALE/YarsRevenge-v5 | **12873.91** | - | - |
| ALE/Zaxxon-v5 | **9523.49** | - | - |

**Legend**: **Bold** = Best score | - = Not tested

</details>

### Running Benchmarks

All games use the same spec file with variable substitution for the environment.

**Remote (recommended)** - cloud GPU via [dstack](https://dstack.ai), auto-syncs to HuggingFace:
```bash
# A2C (10M frames)
source .env && slm-lab run-remote --gpu -s env=ALE/Breakout-v5 -s max_frame=1e7 \
  slm_lab/spec/benchmark_arc/a2c/a2c_atari_arc.yaml a2c_gae_atari_arc train -n breakout-a2c

# PPO (10M frames)
source .env && slm-lab run-remote --gpu -s env=ALE/Breakout-v5 -s max_frame=1e7 \
  slm_lab/spec/benchmark_arc/ppo/ppo_atari_arc.yaml ppo_atari_lam70_arc train -n breakout-ppo

# SAC (2M frames - off-policy, more sample-efficient but slower per frame)
source .env && slm-lab run-remote --gpu -s env=ALE/Breakout-v5 \
  slm_lab/spec/benchmark_arc/sac/sac_atari_arc.yaml sac_atari_arc train -n breakout-sac
```

Remote setup: `cp .env.example .env` then set `HF_TOKEN`. See [Remote Training](../using-slm-lab/remote-training.md) for dstack config.

**Local** - runs on your machine (requires GPU, ~2-3 hours per game):
```bash
slm-lab run -s env=ALE/Breakout-v5 -s max_frame=1e7 slm_lab/spec/benchmark_arc/ppo/ppo_atari_arc.yaml ppo_atari_arc train
```

{% hint style="warning" %}
**GPU required for Atari.** Each game runs 10M frames and takes 2-3 hours on cloud GPU (L4/A10G). Local CPU training is not practical. Cloud GPUs via dstack are faster and often cheaper than running on local hardware.
{% endhint %}

### Download and Replay

```bash
# List Atari experiments (requires HF_REPO=SLM-Lab/benchmark in .env)
source .env && slm-lab list | grep atari

# Download a specific game
source .env && slm-lab pull ppo_atari_arc_breakout

# Replay
slm-lab run slm_lab/spec/benchmark_arc/ppo/ppo_atari_arc.yaml ppo_atari_lam70_arc enjoy@data/ppo_atari_lam70_arc_breakout_*/ppo_atari_lam70_arc_breakout_t0_spec.json
```

---

## Results

| ENV | Score | SPEC_NAME | HF Data |
|-----|-------|-----------|---------|
| ALE/AirRaid-v5 | 7042.84 | ppo_atari_arc | [ppo_atari_arc_airraid_2026_02_13_124015](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_airraid_2026_02_13_124015) |
| | 1832.54 | sac_atari_arc | [sac_atari_arc_airraid_2026_02_17_104002](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_airraid_2026_02_17_104002) |
| | 5067 | a2c_gae_atari_arc | [a2c_gae_atari_airraid_2026_02_01_082446](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_airraid_2026_02_01_082446) |
| ALE/Alien-v5 | 1789.26 | ppo_atari_arc | [ppo_atari_arc_alien_2026_02_13_124017](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_alien_2026_02_13_124017) |
| | 833.53 | sac_atari_arc | [sac_atari_arc_alien_2026_02_15_200940](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_alien_2026_02_15_200940) |
| | 1488 | a2c_gae_atari_arc | [a2c_gae_atari_alien_2026_02_01_000858](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_alien_2026_02_01_000858) |
| ALE/Amidar-v5 | 584.28 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_amidar_2026_02_13_124155](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_amidar_2026_02_13_124155) |
| | 185.45 | sac_atari_arc | [sac_atari_arc_amidar_2026_02_16_042529](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_amidar_2026_02_16_042529) |
| | 330 | a2c_gae_atari_arc | [a2c_gae_atari_amidar_2026_02_01_082251](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_amidar_2026_02_01_082251) |
| ALE/Assault-v5 | 4448.16 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_assault_2026_02_13_124219](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_assault_2026_02_13_124219) |
| | 1009.42 | sac_atari_arc | [sac_atari_arc_assault_2026_02_16_042532](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_assault_2026_02_16_042532) |
| | 1646 | a2c_gae_atari_arc | [a2c_gae_atari_assault_2026_02_01_082252](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_assault_2026_02_01_082252) |
| ALE/Asterix-v5 | 3235.46 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_asterix_2026_02_13_124329](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_asterix_2026_02_13_124329) |
| | 1504.44 | sac_atari_arc | [sac_atari_arc_asterix_2026_02_16_064430](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_asterix_2026_02_16_064430) |
| | 2712 | a2c_gae_atari_arc | [a2c_gae_atari_asterix_2026_02_01_082315](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_asterix_2026_02_01_082315) |
| ALE/Asteroids-v5 | 1577.92 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_asteroids_2026_02_13_171445](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_asteroids_2026_02_13_171445) |
| | 1203.52 | sac_atari_arc | [sac_atari_arc_asteroids_2026_02_16_051747](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_asteroids_2026_02_16_051747) |
| | 2106 | a2c_gae_atari_arc | [a2c_gae_atari_asteroids_2026_02_01_082328](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_asteroids_2026_02_01_082328) |
| ALE/Atlantis-v5 | 848087.19 | ppo_atari_arc | [ppo_atari_arc_atlantis_2026_02_13_171349](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_atlantis_2026_02_13_171349) |
| | 56787.32 | sac_atari_arc | [sac_atari_arc_atlantis_2026_02_17_105837](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_atlantis_2026_02_17_105837) |
| | 873365 | a2c_gae_atari_arc | [a2c_gae_atari_atlantis_2026_02_01_082330](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_atlantis_2026_02_01_082330) |
| ALE/BankHeist-v5 | 1058.25 | ppo_atari_arc | [ppo_atari_arc_bankheist_2026_02_13_230416](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_bankheist_2026_02_13_230416) |
| | 138.43 | sac_atari_arc | [sac_atari_arc_bankheist_2026_02_17_105306](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_bankheist_2026_02_17_105306) |
| | 1099 | a2c_gae_atari_arc | [a2c_gae_atari_bankheist_2026_02_01_082403](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_bankheist_2026_02_01_082403) |
| ALE/BattleZone-v5 | 27176.78 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_battlezone_2026_02_13_171436](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_battlezone_2026_02_13_171436) |
| | 6906.47 | sac_atari_arc | [sac_atari_arc_battlezone_2026_02_17_112313](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_battlezone_2026_02_17_112313) |
| | 2437 | a2c_gae_atari_arc | [a2c_gae_atari_battlezone_2026_02_01_082425](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_battlezone_2026_02_01_082425) |
| ALE/BeamRider-v5 | 2761.75 | ppo_atari_arc | [ppo_atari_arc_beamrider_2026_02_13_171450](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_beamrider_2026_02_13_171450) |
| | 4061.05 | sac_atari_arc | [sac_atari_arc_beamrider_2026_02_17_110505](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_beamrider_2026_02_17_110505) |
| | 2767 | a2c_gae_atari_arc | [a2c_gae_atari_beamrider_2026_02_01_000921](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_beamrider_2026_02_01_000921) |
| ALE/Berzerk-v5 | 835.46 | ppo_atari_arc | [ppo_atari_arc_berzerk_2026_02_13_171449](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_berzerk_2026_02_13_171449) |
| | 313.87 | sac_atari_arc | [sac_atari_arc_berzerk_2026_02_17_105608](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_berzerk_2026_02_17_105608) |
| | 439 | a2c_gae_atari_arc | [a2c_gae_atari_berzerk_2026_02_01_082540](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_berzerk_2026_02_01_082540) |
| ALE/Bowling-v5 | 45.02 | ppo_atari_arc | [ppo_atari_arc_bowling_2026_02_13_230507](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_bowling_2026_02_13_230507) |
| | 26.55 | sac_atari_arc | [sac_atari_arc_bowling_2026_02_18_101223](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_bowling_2026_02_18_101223) |
| | 23.96 | a2c_gae_atari_arc | [a2c_gae_atari_bowling_2026_02_01_082529](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_bowling_2026_02_01_082529) |
| ALE/Boxing-v5 | 92.18 | ppo_atari_arc | [ppo_atari_arc_boxing_2026_02_13_230504](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_boxing_2026_02_13_230504) |
| | 44.03 | sac_atari_arc | [sac_atari_arc_boxing_2026_02_15_201228](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_boxing_2026_02_15_201228) |
| | 1.80 | a2c_gae_atari_arc | [a2c_gae_atari_boxing_2026_02_01_082539](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_boxing_2026_02_01_082539) |
| ALE/Breakout-v5 | 326.47 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_breakout_2026_02_13_230455](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_breakout_2026_02_13_230455) |
| | 20.23 | sac_atari_arc | [sac_atari_arc_breakout_2026_02_15_201235](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_breakout_2026_02_15_201235) |
| | 273 | a2c_gae_atari_arc | [a2c_gae_atari_breakout_2026_01_31_213610](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_breakout_2026_01_31_213610) |
| ALE/Carnival-v5 | 3912.59 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_carnival_2026_02_13_230438](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_carnival_2026_02_13_230438) |
| | 3501.37 | sac_atari_arc | [sac_atari_arc_carnival_2026_02_17_105834](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_carnival_2026_02_17_105834) |
| | 2170 | a2c_gae_atari_arc | [a2c_gae_atari_carnival_2026_02_01_082726](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_carnival_2026_02_01_082726) |
| ALE/Centipede-v5 | 4780.75 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_centipede_2026_02_13_230434](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_centipede_2026_02_13_230434) |
| | 2255.45 | sac_atari_arc | [sac_atari_arc_centipede_2026_02_18_101425](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_centipede_2026_02_18_101425) |
| | 1382 | a2c_gae_atari_arc | [a2c_gae_atari_centipede_2026_02_01_082643](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_centipede_2026_02_01_082643) |
| ALE/ChopperCommand-v5 | 5391.30 | ppo_atari_arc | [ppo_atari_arc_choppercommand_2026_02_13_230448](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_choppercommand_2026_02_13_230448) |
| | 1036.91 | sac_atari_arc | [sac_atari_arc_choppercommand_2026_02_17_110523](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_choppercommand_2026_02_17_110523) |
| | 2446 | a2c_gae_atari_arc | [a2c_gae_atari_choppercommand_2026_02_01_082626](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_choppercommand_2026_02_01_082626) |
| ALE/CrazyClimber-v5 | 112094.03 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_crazyclimber_2026_02_13_230445](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_crazyclimber_2026_02_13_230445) |
| | 75712.12 | sac_atari_arc | [sac_atari_arc_crazyclimber_2026_02_15_201349](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_crazyclimber_2026_02_15_201349) |
| | 96943 | a2c_gae_atari_arc | [a2c_gae_atari_crazyclimber_2026_02_01_082625](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_crazyclimber_2026_02_01_082625) |
| ALE/Defender-v5 | 47894.69 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_defender_2026_02_14_023317](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_defender_2026_02_14_023317) |
| | 4386.79 | sac_atari_arc | [sac_atari_arc_defender_2026_02_18_101518](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_defender_2026_02_18_101518) |
| | 33149 | a2c_gae_atari_arc | [a2c_gae_atari_defender_2026_02_01_082658](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_defender_2026_02_01_082658) |
| ALE/DemonAttack-v5 | 19370.38 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_demonattack_2026_02_14_023650](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_demonattack_2026_02_14_023650) |
| | 4555.58 | sac_atari_arc | [sac_atari_arc_demonattack_2026_02_18_101610](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_demonattack_2026_02_18_101610) |
| | 2962 | a2c_gae_atari_arc | [a2c_gae_atari_demonattack_2026_02_01_082717](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_demonattack_2026_02_01_082717) |
| ALE/DoubleDunk-v5 | -3.03 | ppo_atari_arc | [ppo_atari_arc_doubledunk_2026_02_14_043639](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_doubledunk_2026_02_14_043639) |
| | -18.65 | sac_atari_arc | [sac_atari_arc_doubledunk_2026_02_17_160707](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_doubledunk_2026_02_17_160707) |
| | -1.69 | a2c_gae_atari_arc | [a2c_gae_atari_doubledunk_2026_02_01_082901](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_doubledunk_2026_02_01_082901) |
| ALE/Enduro-v5 | 986.46 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_enduro_2026_02_11_101739](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_enduro_2026_02_11_101739) |
| | 45.80 | sac_atari_arc | [sac_atari_arc_enduro_2026_02_17_160716](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_enduro_2026_02_17_160716) |
| | 681 | a2c_gae_atari_arc | [a2c_gae_atari_enduro_2026_02_01_001123](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_enduro_2026_02_01_001123) |
| ALE/FishingDerby-v5 | 25.71 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_fishingderby_2026_02_14_024158](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_fishingderby_2026_02_14_024158) |
| | -75.82 | sac_atari_arc | [sac_atari_arc_fishingderby_2026_02_17_160848](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_fishingderby_2026_02_17_160848) |
| | -16.38 | a2c_gae_atari_arc | [a2c_gae_atari_fishingderby_2026_02_01_082906](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_fishingderby_2026_02_01_082906) |
| ALE/Freeway-v5 | 32.42 | ppo_atari_arc | [ppo_atari_arc_freeway_2026_02_14_023359](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_freeway_2026_02_14_023359) |
| | 0.00 | sac_atari_arc | [sac_atari_arc_freeway_2026_02_17_161324](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_freeway_2026_02_17_161324) |
| | 23.13 | a2c_gae_atari_arc | [a2c_gae_atari_freeway_2026_02_01_082931](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_freeway_2026_02_01_082931) |
| ALE/Frostbite-v5 | 284.07 | ppo_atari_arc | [ppo_atari_arc_frostbite_2026_02_14_024247](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_frostbite_2026_02_14_024247) |
| | 355.80 | sac_atari_arc | [sac_atari_arc_frostbite_2026_02_17_160759](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_frostbite_2026_02_17_160759) |
| | 266 | a2c_gae_atari_arc | [a2c_gae_atari_frostbite_2026_02_01_082915](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_frostbite_2026_02_01_082915) |
| ALE/Gopher-v5 | 6500.38 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_gopher_2026_02_14_024237](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_gopher_2026_02_14_024237) |
| | 1608.59 | sac_atari_arc | [sac_atari_arc_gopher_2026_02_17_161047](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_gopher_2026_02_17_161047) |
| | 984 | a2c_gae_atari_arc | [a2c_gae_atari_gopher_2026_02_01_133323](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_gopher_2026_02_01_133323) |
| ALE/Gravitar-v5 | 602.58 | ppo_atari_arc | [ppo_atari_arc_gravitar_2026_02_14_075743](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_gravitar_2026_02_14_075743) |
| | 233.02 | sac_atari_arc | [sac_atari_arc_gravitar_2026_02_17_160858](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_gravitar_2026_02_17_160858) |
| | 270 | a2c_gae_atari_arc | [a2c_gae_atari_gravitar_2026_02_01_133244](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_gravitar_2026_02_01_133244) |
| ALE/Hero-v5 | 22477.89 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_hero_2026_02_15_232615](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_hero_2026_02_15_232615) |
| | 4873.09 | sac_atari_arc | [sac_atari_arc_hero_2026_02_17_161420](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_hero_2026_02_17_161420) |
| | 18680 | a2c_gae_atari_arc | [a2c_gae_atari_hero_2026_02_01_175903](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_hero_2026_02_01_175903) |
| ALE/IceHockey-v5 | -4.05 | ppo_atari_arc | [ppo_atari_arc_icehockey_2026_02_14_231829](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_icehockey_2026_02_14_231829) |
| | -19.78 | sac_atari_arc | [sac_atari_arc_icehockey_2026_02_18_101834](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_icehockey_2026_02_18_101834) |
| | -5.92 | a2c_gae_atari_arc | [a2c_gae_atari_icehockey_2026_02_01_175745](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_icehockey_2026_02_01_175745) |
| ALE/Jamesbond-v5 | 710.98 | ppo_atari_arc | [ppo_atari_arc_jamesbond_2026_02_14_080649](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_jamesbond_2026_02_14_080649) |
| | 328.27 | sac_atari_arc | [sac_atari_arc_jamesbond_2026_02_17_220305](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_jamesbond_2026_02_17_220305) |
| | 460 | a2c_gae_atari_arc | [a2c_gae_atari_jamesbond_2026_02_01_175945](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_jamesbond_2026_02_01_175945) |
| ALE/JourneyEscape-v5 | -1248.98 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_journeyescape_2026_02_14_080656](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_journeyescape_2026_02_14_080656) |
| | -3268.80 | sac_atari_arc | [sac_atari_arc_journeyescape_2026_02_17_215843](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_journeyescape_2026_02_17_215843) |
| | -965 | a2c_gae_atari_arc | [a2c_gae_atari_journeyescape_2026_02_01_084415](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_journeyescape_2026_02_01_084415) |
| ALE/Kangaroo-v5 | 10660.35 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_kangaroo_2026_02_16_030656](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_kangaroo_2026_02_16_030656) |
| | 2990.74 | sac_atari_arc | [sac_atari_arc_kangaroo_2026_02_17_220652](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_kangaroo_2026_02_17_220652) |
| | 322 | a2c_gae_atari_arc | [a2c_gae_atari_kangaroo_2026_02_01_084415](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_kangaroo_2026_02_01_084415) |
| ALE/Krull-v5 | 7874.33 | ppo_atari_arc | [ppo_atari_arc_krull_2026_02_14_080657](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_krull_2026_02_14_080657) |
| | 6630.02 | sac_atari_arc | [sac_atari_arc_krull_2026_02_17_221656](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_krull_2026_02_17_221656) |
| | 7519 | a2c_gae_atari_arc | [a2c_gae_atari_krull_2026_02_01_084420](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_krull_2026_02_01_084420) |
| ALE/KungFuMaster-v5 | 28128.04 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_kungfumaster_2026_02_14_080730](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_kungfumaster_2026_02_14_080730) |
| | 9932.72 | sac_atari_arc | [sac_atari_arc_kungfumaster_2026_02_17_221024](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_kungfumaster_2026_02_17_221024) |
| | 23006 | a2c_gae_atari_arc | [a2c_gae_atari_kungfumaster_2026_02_01_085101](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_kungfumaster_2026_02_01_085101) |
| ALE/MsPacman-v5 | 2330.74 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_mspacman_2026_02_14_102435](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_mspacman_2026_02_14_102435) |
| | 1336.96 | sac_atari_arc | [sac_atari_arc_mspacman_2026_02_17_221523](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_mspacman_2026_02_17_221523) |
| | 2110 | a2c_gae_atari_arc | [a2c_gae_atari_mspacman_2026_02_01_001100](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_mspacman_2026_02_01_001100) |
| ALE/NameThisGame-v5 | 6879.23 | ppo_atari_arc | [ppo_atari_arc_namethisgame_2026_02_14_103319](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_namethisgame_2026_02_14_103319) |
| | 3992.71 | sac_atari_arc | [sac_atari_arc_namethisgame_2026_02_17_220905](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_namethisgame_2026_02_17_220905) |
| | 5412 | a2c_gae_atari_arc | [a2c_gae_atari_namethisgame_2026_02_01_132733](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_namethisgame_2026_02_01_132733) |
| ALE/Phoenix-v5 | 13923.26 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_phoenix_2026_02_14_102636](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_phoenix_2026_02_14_102636) |
| | 3958.46 | sac_atari_arc | [sac_atari_arc_phoenix_2026_02_17_222102](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_phoenix_2026_02_17_222102) |
| | 5635 | a2c_gae_atari_arc | [a2c_gae_atari_phoenix_2026_02_01_085101](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_phoenix_2026_02_01_085101) |
| ALE/Pong-v5 | 16.69 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_pong_2026_02_14_103722](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_pong_2026_02_14_103722) |
| | 10.89 | sac_atari_arc | [sac_atari_arc_pong_2026_02_17_160429](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_pong_2026_02_17_160429) |
| | 10.17 | a2c_gae_atari_arc | [a2c_gae_atari_pong_2026_01_31_213635](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_pong_2026_01_31_213635) |
| ALE/Pooyan-v5 | 5308.66 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_pooyan_2026_02_14_114730](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_pooyan_2026_02_14_114730) |
| | 2530.78 | sac_atari_arc | [sac_atari_arc_pooyan_2026_02_17_220346](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_pooyan_2026_02_17_220346) |
| | 2997 | a2c_gae_atari_arc | [a2c_gae_atari_pooyan_2026_02_01_132748](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_pooyan_2026_02_01_132748) |
| ALE/Qbert-v5 | 15460.48 | ppo_atari_arc | [ppo_atari_arc_qbert_2026_02_14_120409](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_qbert_2026_02_14_120409) |
| | 3331.98 | sac_atari_arc | [sac_atari_arc_qbert_2026_02_17_223117](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_qbert_2026_02_17_223117) |
| | 12619 | a2c_gae_atari_arc | [a2c_gae_atari_qbert_2026_01_31_213720](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_qbert_2026_01_31_213720) |
| ALE/Riverraid-v5 | 9599.75 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_riverraid_2026_02_14_124700](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_riverraid_2026_02_14_124700) |
| | 4744.95 | sac_atari_arc | [sac_atari_arc_riverraid_2026_02_18_014310](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_riverraid_2026_02_18_014310) |
| | 6558 | a2c_gae_atari_arc | [a2c_gae_atari_riverraid_2026_02_01_132507](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_riverraid_2026_02_01_132507) |
| ALE/RoadRunner-v5 | 37980.95 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_roadrunner_2026_02_14_124844](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_roadrunner_2026_02_14_124844) |
| | 25975.39 | sac_atari_arc | [sac_atari_arc_roadrunner_2026_02_18_015052](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_roadrunner_2026_02_18_015052) |
| | 29810 | a2c_gae_atari_arc | [a2c_gae_atari_roadrunner_2026_02_01_132509](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_roadrunner_2026_02_01_132509) |
| ALE/Robotank-v5 | 21.04 | ppo_atari_arc | [ppo_atari_arc_robotank_2026_02_14_124751](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_robotank_2026_02_14_124751) |
| | 9.01 | sac_atari_arc | [sac_atari_arc_robotank_2026_02_18_032313](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_robotank_2026_02_18_032313) |
| | 2.80 | a2c_gae_atari_arc | [a2c_gae_atari_robotank_2026_02_01_132434](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_robotank_2026_02_01_132434) |
| ALE/Seaquest-v5 | 1775.14 | ppo_atari_arc | [ppo_atari_arc_seaquest_2026_02_11_095444](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_seaquest_2026_02_11_095444) |
| | 1565.44 | sac_atari_arc | [sac_atari_arc_seaquest_2026_02_18_020822](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_seaquest_2026_02_18_020822) |
| | 850 | a2c_gae_atari_arc | [a2c_gae_atari_seaquest_2026_02_01_001001](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_seaquest_2026_02_01_001001) |
| ALE/Skiing-v5 | -28217.28 | ppo_atari_arc | [ppo_atari_arc_skiing_2026_02_14_174807](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_skiing_2026_02_14_174807) |
| | -17464.22 | sac_atari_arc | [sac_atari_arc_skiing_2026_02_18_024444](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_skiing_2026_02_18_024444) |
| | -14235 | a2c_gae_atari_arc | [a2c_gae_atari_skiing_2026_02_01_132451](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_skiing_2026_02_01_132451) |
| ALE/Solaris-v5 | 2212.78 | ppo_atari_arc | [ppo_atari_arc_solaris_2026_02_14_124751](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_solaris_2026_02_14_124751) |
| | 1803.74 | sac_atari_arc | [sac_atari_arc_solaris_2026_02_18_031943](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_solaris_2026_02_18_031943) |
| | 2224 | a2c_gae_atari_arc | [a2c_gae_atari_solaris_2026_02_01_212137](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_solaris_2026_02_01_212137) |
| ALE/SpaceInvaders-v5 | 892.49 | ppo_atari_arc | [ppo_atari_arc_spaceinvaders_2026_02_14_131114](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_spaceinvaders_2026_02_14_131114) |
| | 507.33 | sac_atari_arc | [sac_atari_arc_spaceinvaders_2026_02_18_033139](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_spaceinvaders_2026_02_18_033139) |
| | 784 | a2c_gae_atari_arc | [a2c_gae_atari_spaceinvaders_2026_02_01_000950](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_spaceinvaders_2026_02_01_000950) |
| ALE/StarGunner-v5 | 49328.73 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_stargunner_2026_02_14_131149](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_stargunner_2026_02_14_131149) |
| | 4295.97 | sac_atari_arc | [sac_atari_arc_stargunner_2026_02_18_033151](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_stargunner_2026_02_18_033151) |
| | 8665 | a2c_gae_atari_arc | [a2c_gae_atari_stargunner_2026_02_01_132406](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_stargunner_2026_02_01_132406) |
| ALE/Surround-v5 | -4.47 | ppo_atari_arc | [ppo_atari_arc_surround_2026_02_14_132941](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_surround_2026_02_14_132941) |
| | -9.87 | sac_atari_arc | [sac_atari_arc_surround_2026_02_18_034423](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_surround_2026_02_18_034423) |
| | -9.72 | a2c_gae_atari_arc | [a2c_gae_atari_surround_2026_02_01_132215](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_surround_2026_02_01_132215) |
| ALE/Tennis-v5 | -12.27 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_tennis_2026_02_14_173639](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_tennis_2026_02_14_173639) |
| | -397.44 | sac_atari_arc | [sac_atari_arc_tennis_2026_02_18_032540](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_tennis_2026_02_18_032540) |
| | -2873 | a2c_gae_atari_arc | [a2c_gae_atari_tennis_2026_02_01_175829](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_tennis_2026_02_01_175829) |
| ALE/TimePilot-v5 | 4432.73 | ppo_atari_arc | [ppo_atari_arc_timepilot_2026_02_14_173642](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_timepilot_2026_02_14_173642) |
| | 3164.97 | sac_atari_arc | [sac_atari_arc_timepilot_2026_02_18_102038](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_timepilot_2026_02_18_102038) |
| | 3376 | a2c_gae_atari_arc | [a2c_gae_atari_timepilot_2026_02_01_175930](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_timepilot_2026_02_01_175930) |
| ALE/Tutankham-v5 | 210.87 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_tutankham_2026_02_14_173722](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_tutankham_2026_02_14_173722) |
| | 147.25 | sac_atari_arc | [sac_atari_arc_tutankham_2026_02_18_102729](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_tutankham_2026_02_18_102729) |
| | 167 | a2c_gae_atari_arc | [a2c_gae_atari_tutankham_2026_02_01_132347](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_tutankham_2026_02_01_132347) |
| ALE/UpNDown-v5 | 147168.80 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_upndown_2026_02_15_232448](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_upndown_2026_02_15_232448) |
| | 3351.89 | sac_atari_arc | [sac_atari_arc_upndown_2026_02_18_135442](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_upndown_2026_02_18_135442) |
| | 57099 | a2c_gae_atari_arc | [a2c_gae_atari_upndown_2026_02_01_132435](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_upndown_2026_02_01_132435) |
| ALE/VideoPinball-v5 | 38370.30 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_videopinball_2026_02_14_173728](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_videopinball_2026_02_14_173728) |
| | 21088.68 | sac_atari_arc | [sac_atari_arc_videopinball_2026_02_18_141245](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_videopinball_2026_02_18_141245) |
| | 25310 | a2c_gae_atari_arc | [a2c_gae_atari_videopinball_2026_02_01_083457](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_videopinball_2026_02_01_083457) |
| ALE/WizardOfWor-v5 | 6100.42 | ppo_atari_arc | [ppo_atari_arc_wizardofwor_2026_02_14_173945](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_wizardofwor_2026_02_14_173945) |
| | 1241.92 | sac_atari_arc | [sac_atari_arc_wizardofwor_2026_02_18_140750](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_wizardofwor_2026_02_18_140750) |
| | 2682 | a2c_gae_atari_arc | [a2c_gae_atari_wizardofwor_2026_02_01_132449](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_wizardofwor_2026_02_01_132449) |
| ALE/YarsRevenge-v5 | 12873.91 | ppo_atari_arc | [ppo_atari_arc_yarsrevenge_2026_02_14_174019](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_yarsrevenge_2026_02_14_174019) |
| | 13710.18 | sac_atari_arc | [sac_atari_arc_yarsrevenge_2026_02_18_134921](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_yarsrevenge_2026_02_18_134921) |
| | 24371 | a2c_gae_atari_arc | [a2c_gae_atari_yarsrevenge_2026_02_01_132224](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_yarsrevenge_2026_02_01_132224) |
| ALE/Zaxxon-v5 | 9523.49 | ppo_atari_arc | [ppo_atari_arc_zaxxon_2026_02_14_174806](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_zaxxon_2026_02_14_174806) |
| | 3205.98 | sac_atari_arc | [sac_atari_arc_zaxxon_2026_02_18_135502](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_zaxxon_2026_02_18_135502) |
| | 29.46 | a2c_gae_atari_arc | [a2c_gae_atari_zaxxon_2026_02_01_131758](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_zaxxon_2026_02_01_131758) |

**Skipped**: Adventure, MontezumaRevenge, Pitfall, PrivateEye, Venture (hard exploration), ElevatorAction (deprecated env)

### Training Curves

A2C vs PPO vs SAC mean returns (moving average) vs training frames. Shaded regions show standard deviation across 4 sessions.

| | | |
|:---:|:---:|:---:|
| ![AirRaid](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/AirRaid-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Alien](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Alien-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Amidar](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Amidar-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![Assault](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Assault-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Asterix](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Asterix-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Asteroids](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Asteroids-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![Atlantis](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Atlantis-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![BankHeist](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/BankHeist-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![BattleZone](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/BattleZone-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![BeamRider](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/BeamRider-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Berzerk](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Berzerk-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Bowling](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Bowling-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![Boxing](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Boxing-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Breakout](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Breakout-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Carnival](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Carnival-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![Centipede](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Centipede-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![ChopperCommand](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/ChopperCommand-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![CrazyClimber](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/CrazyClimber-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![Defender](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Defender-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![DemonAttack](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/DemonAttack-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![DoubleDunk](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/DoubleDunk-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![Enduro](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Enduro-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![FishingDerby](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/FishingDerby-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Freeway](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Freeway-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![Frostbite](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Frostbite-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Gopher](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Gopher-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Gravitar](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Gravitar-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![Hero](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Hero-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![IceHockey](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/IceHockey-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Jamesbond](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Jamesbond-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![JourneyEscape](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/JourneyEscape-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Kangaroo](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Kangaroo-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Krull](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Krull-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![KungFuMaster](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/KungFuMaster-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![MsPacman](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/MsPacman-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![NameThisGame](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/NameThisGame-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![Phoenix](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Phoenix-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Pong](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Pong-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Pooyan](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Pooyan-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![Qbert](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Qbert-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Riverraid](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Riverraid-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![RoadRunner](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/RoadRunner-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![Robotank](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Robotank-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Seaquest](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Seaquest-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Skiing](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Skiing-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![Solaris](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Solaris-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![SpaceInvaders](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/SpaceInvaders-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![StarGunner](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/StarGunner-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![Surround](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Surround-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Tennis](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Tennis-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![TimePilot](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/TimePilot-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![Tutankham](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Tutankham-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![UpNDown](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/UpNDown-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![VideoPinball](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/VideoPinball-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |
| ![WizardOfWor](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/WizardOfWor-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![YarsRevenge](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/YarsRevenge-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) | ![Zaxxon](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Zaxxon-v5_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260218) |

---

## Historical Results (v4)

<details>
<summary><b>OpenAI Gym Atari Results (v4)</b> - click to expand</summary>

{% hint style="warning" %}
**Deprecated Environments:** These v4 results used OpenAI Gym `NoFrameskip-v4` environments (no sticky actions). Gymnasium ALE v5 environments are harder due to sticky action probability. Results are not directly comparable.
{% endhint %}

* [Upload PR #427](https://github.com/kengz/SLM-Lab/pull/427)
* [Google Drive data: DQN](https://drive.google.com/file/d/1taFdNmrL535zJ4V7wRNORwkH_mgSoiYz/view?usp=sharing)
* [Google Drive data: DDQN+PER](https://drive.google.com/file/d/1PrMn-qvh51szKm-AbFdBphXDicDoRO0d/view?usp=sharing)
* [Google Drive data: A2C (GAE)](https://drive.google.com/file/d/10T7ehim0cGfWxWZkMHjNG5m1sph0OuHb/view?usp=sharing)
* [Google Drive data: A2C (n-step)](https://drive.google.com/file/d/17v_PVkxucFtVzm2MW9nWAQv7mn76Ukc3/view?usp=sharing)
* [Google Drive data: PPO](https://drive.google.com/file/d/1CQdF_jBZKNL58cDIMvIlvLLY-LPj01nv/view?usp=sharing)
* [Google Drive data: all Atari Graphs](https://drive.google.com/file/d/11g9pC-MEzIuRYOoqvyqIxn4GtXpIvdau/view?usp=sharing)

|      Env. \ Alg. |   DQN   |  DDQN+PER  |  A2C (GAE)  | A2C (n-step) |     PPO    |
| ---------------: | :-----: | :--------: | :---------: | :----------: | :--------: |
|        Adventure |  -0.94  |    -0.92   |    -0.77    |     -0.85    |  **-0.3**  |
|          AirRaid |   1876  |    3974    |   **4202**  |     3557     |    4028    |
|            Alien |   822   |    1574    |     1519    |   **1627**   |    1413    |
|           Amidar |  90.95  |     431    |     577     |      418     |   **795**  |
|          Assault |   1392  |    2567    |     3366    |     3312     |  **3619**  |
|          Asterix |   1253  |  **6866**  |     5559    |     5223     |    6132    |
|        Asteroids |   439   |     426    |   **2951**  |     2147     |    2186    |
|         Atlantis |  68679  |   644810   | **2747371** |    2259733   |   2148077  |
|        BankHeist |   131   |     623    |     855     |     1170     |  **1183**  |
|       BattleZone |   6564  |    6395    |     4336    |     4533     |  **13649** |
|        BeamRider |   2799  |  **5870**  |     2659    |     4139     |    4299    |
|          Berzerk |   319   |     401    |   **1073**  |      763     |     860    |
|          Bowling |  30.29  |  **39.5**  |    24.51    |     23.75    |    31.64   |
|           Boxing |  72.11  |    90.98   |     1.57    |     1.26     |  **96.53** |
|         Breakout |  80.88  |     182    |     377     |      398     |   **443**  |
|         Carnival |   4280  |  **4773**  |     2473    |     1827     |    4566    |
|        Centipede |   1899  |    2153    |     3909    |     4202     |  **5003**  |
|   ChopperCommand |   1083  |  **4020**  |     3043    |     1280     |    3357    |
|     CrazyClimber |  46984  |    88814   |    106256   |    109998    | **116820** |
|         Defender |  281999 |   313018   |  **665609** |    657823    |   534639   |
|      DemonAttack |   1705  |    19856   |    23779    |     19615    | **121172** |
|       DoubleDunk |  -21.44 |   -22.38   |  **-5.15**  |     -13.3    |    -6.01   |
|   ElevatorAction |  32.62  |    17.91   |   **9966**  |     8818     |    6471    |
|           Enduro |   437   |     959    |     787     |      0.0     |  **1926**  |
|     FishingDerby |  -88.14 |    -1.7    |    16.54    |     1.65     |  **36.03** |
|          Freeway |  24.46  |    30.49   |    30.97    |      0.0     |  **32.11** |
|        Frostbite |   98.8  |  **2497**  |     277     |      261     |    1062    |
|           Gopher |   1095  |  **7562**  |     929     |     1545     |    2933    |
|         Gravitar |  87.34  |     258    |     313     |    **433**   |     223    |
|             Hero |   1051  |    12579   |    16502    |   **19322**  |    17412   |
|        IceHockey |  -14.96 |   -14.24   |  **-5.79**  |     -6.06    |    -6.43   |
|        Jamesbond |  44.87  |   **702**  |     521     |      453     |     561    |
|    JourneyEscape |  -4818  |    -2003   |   **-921**  |     -2032    |    -1094   |
|         Kangaroo |   1965  |  **8897**  |    67.62    |      554     |    4989    |
|            Krull |   5522  |    6650    |     7785    |     6642     |  **8477**  |
|     KungFuMaster |   2288  |    16547   |    31199    |     25554    |  **34523** |
| MontezumaRevenge |   0.0   |    0.02    |     0.08    |     0.19     |  **1.08**  |
|         MsPacman |   1175  |    2215    |     1965    |     2158     |  **2350**  |
|     NameThisGame |   3915  |    4474    |     5178    |     5795     |  **6386**  |
|          Phoenix |   2909  |    8179    |    16345    |     13586    |  **30504** |
|          Pitfall |  -68.83 |   -73.65   |     -101    |  **-31.13**  |   -35.93   |
|             Pong |  18.48  |    20.5    |    19.31    |     19.56    |  **20.58** |
|           Pooyan |   1958  |    2741    |     2862    |     2531     |  **6799**  |
|       PrivateEye | **784** |     303    |    93.22    |     78.07    |    50.12   |
|            Qbert |   5494  |    11426   |    12405    |   **13590**  |    13460   |
|        Riverraid |   953   |  **10492** |     8308    |     7565     |    9636    |
|       RoadRunner |  15237  |    29047   |    30152    |     31030    |  **32956** |
|         Robotank |   3.43  |  **9.05**  |     2.98    |     2.27     |    2.27    |
|         Seaquest |   1185  |  **4405**  |     1070    |     1684     |    1715    |
|           Skiing |  -14094 | **-12883** |    -19481   |    -14234    |   -24713   |
|          Solaris |   612   |    1396    |     2115    |   **2236**   |    1892    |
|    SpaceInvaders |   451   |     670    |     733     |      750     |   **797**  |
|       StarGunner |   3565  |    38238   |    44816    |     48410    |  **60579** |
|           Tennis |  -23.78 | **-10.33** |    -22.42   |    -19.06    |   -11.52   |
|        TimePilot |   2819  |    1884    |     3331    |     3440     |  **4398**  |
|        Tutankham |  35.03  |     159    |     161     |      175     |   **211**  |
|          UpNDown |   2043  |    11632   |    89769    |     18878    | **262208** |
|          Venture |   4.56  |    9.61    |     0.0     |      0.0     |  **11.84** |
|     VideoPinball |   8056  |  **79730** |    35371    |     40423    |    58096   |
|      WizardOfWor |   869   |     328    |     1516    |     1247     |  **4283**  |
|      YarsRevenge |   5816  |    15698   |  **27097**  |     11742    |    10114   |
|           Zaxxon |   442   |    54.28   |    64.72    |     24.7     |   **641**  |

> The table above presents results for 62 Atari games. All agents were trained for 10M frames (40M including skipped frames). Reported results are the episode score at the end of training, averaged over the previous 100 evaluation checkpoints with each checkpoint averaged over 4 Sessions.

</details>
