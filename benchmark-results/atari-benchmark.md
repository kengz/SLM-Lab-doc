# Atari Environment Benchmark 👾

## A2C & PPO Atari Results (v5)

SLM Lab v5 validates A2C and PPO on [ALE (Arcade Learning Environment)](https://ale.farama.org/environments/) environments. The ALE provides 50+ classic Atari 2600 games as standardized RL benchmarks.

| MsPacman | Breakout | Qbert | BeamRider |
|:---:|:---:|:---:|:---:|
| ![MsPacman](https://user-images.githubusercontent.com/8209263/63994685-5cb30d00-caaa-11e9-8f35-78e29a7d60f5.gif) | ![Breakout](https://user-images.githubusercontent.com/8209263/63994695-650b4800-caaa-11e9-9982-2462738caa45.gif) | ![Qbert](https://user-images.githubusercontent.com/8209263/63994672-54f36880-caaa-11e9-9757-7780725b53af.gif) | ![BeamRider](https://user-images.githubusercontent.com/8209263/63994698-689ecf00-caaa-11e9-991f-0a5e9c2f5804.gif) |

**54 games tested** with all results available on [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark).

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

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 10000

**Algorithm Specs** (all use Nature CNN [32,64,64] + 512fc):
- **DDQN+PER**: Skipped - off-policy variants ~6x slower (~230 fps vs ~1500 fps), not cost effective at 10M frames
- **A2C**: [a2c_gae_atari.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/a2c/a2c_gae_atari.json) - RMSprop (lr=7e-4), training_frequency=32
- **PPO**: [ppo_atari.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_atari.json) - AdamW (lr=2.5e-4), minibatch=256, horizon=128, epochs=4

**Environment:** Gymnasium ALE v5 with `life_loss_info=true`, sticky actions (`repeat_action_probability=0.25`)

### PPO Lambda Variants

Different games benefit from different lambda values for GAE. All variants use the same spec file:

| SPEC_NAME | Lambda | Best for |
|-----------|--------|----------|
| ppo_atari | 0.95 | Strategic games (default) |
| ppo_atari_lam85 | 0.85 | Mixed games |
| ppo_atari_lam70 | 0.70 | Action games |

### Results

Table shows best result per game for each algorithm. PPO results show best lambda variant.

| Game | A2C Score | PPO Score | PPO Variant |
|------|-----------|-----------|-------------|
| ALE/AirRaid-v5 | 5067 | **8245** | ppo_atari |
| ALE/Alien-v5 | **1488** | 1453 | ppo_atari |
| ALE/Amidar-v5 | 330 | **580** | ppo_atari_lam85 |
| ALE/Assault-v5 | 1646 | **4293** | ppo_atari_lam85 |
| ALE/Asterix-v5 | 2712 | **3482** | ppo_atari_lam85 |
| ALE/Asteroids-v5 | **2106** | 1554 | ppo_atari_lam85 |
| ALE/Atlantis-v5 | **873365** | 792886 | ppo_atari |
| ALE/BankHeist-v5 | **1099** | 1045 | ppo_atari |
| ALE/BattleZone-v5 | 2437 | **26383** | ppo_atari_lam85 |
| ALE/BeamRider-v5 | **2767** | 2765 | ppo_atari |
| ALE/Berzerk-v5 | 439 | **1072** | ppo_atari |
| ALE/Bowling-v5 | 23.96 | **46.45** | ppo_atari |
| ALE/Boxing-v5 | 1.80 | **91.17** | ppo_atari |
| ALE/Breakout-v5 | 273 | **327** | ppo_atari_lam70 |
| ALE/Carnival-v5 | 2170 | **3967** | ppo_atari_lam70 |
| ALE/Centipede-v5 | 1382 | **4915** | ppo_atari_lam70 |
| ALE/ChopperCommand-v5 | 2446 | **5355** | ppo_atari |
| ALE/CrazyClimber-v5 | 96943 | **107370** | ppo_atari_lam85 |
| ALE/Defender-v5 | 33149 | **51439** | ppo_atari_lam70 |
| ALE/DemonAttack-v5 | 2962 | **16558** | ppo_atari_lam70 |
| ALE/DoubleDunk-v5 | **-1.69** | -2.38 | ppo_atari |
| ALE/ElevatorAction-v5 | 731 | **5446** | ppo_atari |
| ALE/Enduro-v5 | 681 | **898** | ppo_atari_lam85 |
| ALE/FishingDerby-v5 | -16.38 | **27.10** | ppo_atari_lam85 |
| ALE/Freeway-v5 | 23.13 | **31.30** | ppo_atari |
| ALE/Frostbite-v5 | 266 | **301** | ppo_atari |
| ALE/Gopher-v5 | 984 | **6508** | ppo_atari_lam70 |
| ALE/Gravitar-v5 | 270 | **599** | ppo_atari |
| ALE/Hero-v5 | 18680 | **28238** | ppo_atari_lam85 |
| ALE/IceHockey-v5 | **-5.92** | -3.93 | ppo_atari |
| ALE/Jamesbond-v5 | 460 | **662** | ppo_atari |
| ALE/JourneyEscape-v5 | **-965** | -1252 | ppo_atari_lam85 |
| ALE/Kangaroo-v5 | 322 | **9912** | ppo_atari_lam85 |
| ALE/Krull-v5 | 7519 | **7841** | ppo_atari |
| ALE/KungFuMaster-v5 | 23006 | **29068** | ppo_atari_lam70 |
| ALE/MsPacman-v5 | 2110 | **2372** | ppo_atari_lam85 |
| ALE/NameThisGame-v5 | 5412 | **5993** | ppo_atari |
| ALE/Phoenix-v5 | 5635 | **15659** | ppo_atari_lam70 |
| ALE/Pong-v5 | 10.17 | **16.91** | ppo_atari_lam85 |
| ALE/Pooyan-v5 | 2997 | **5716** | ppo_atari_lam70 |
| ALE/Qbert-v5 | 12619 | **15094** | ppo_atari |
| ALE/Riverraid-v5 | 6558 | **9428** | ppo_atari_lam85 |
| ALE/RoadRunner-v5 | 29810 | **37015** | ppo_atari_lam85 |
| ALE/Robotank-v5 | 2.80 | **20.07** | ppo_atari |
| ALE/Seaquest-v5 | 850 | **1796** | ppo_atari |
| ALE/Skiing-v5 | **-14235** | -19340 | ppo_atari |
| ALE/Solaris-v5 | - | **2094** | ppo_atari |
| ALE/SpaceInvaders-v5 | **784** | 726 | ppo_atari |
| ALE/StarGunner-v5 | 8665 | **47495** | ppo_atari_lam70 |
| ALE/Surround-v5 | -9.72 | **-2.52** | ppo_atari |
| ALE/Tennis-v5 | -2873 | **-4.41** | ppo_atari_lam85 |
| ALE/TimePilot-v5 | 3376 | **4668** | ppo_atari |
| ALE/Tutankham-v5 | 167 | **217** | ppo_atari_lam85 |
| ALE/UpNDown-v5 | 57099 | **182472** | ppo_atari |
| ALE/VideoPinball-v5 | 25310 | **56746** | ppo_atari_lam70 |
| ALE/WizardOfWor-v5 | 2682 | **5814** | ppo_atari |
| ALE/YarsRevenge-v5 | **24371** | 17120 | ppo_atari |
| ALE/Zaxxon-v5 | 29.46 | **10756** | ppo_atari |

**Bold** = best score between A2C and PPO. `-` = not tested.

**Skipped** (hard exploration): Adventure, MontezumaRevenge, Pitfall, PrivateEye, Venture

### Training Curves

Multi-trial comparison plots showing A2C vs PPO mean returns (moving average) vs training frames. Shaded regions show standard deviation across 4 sessions.

| | | |
|:---:|:---:|:---:|
| ![AirRaid](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/AirRaid_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Alien](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Alien_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Amidar](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Amidar_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Assault](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Assault_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Asterix](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Asterix_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Asteroids](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Asteroids_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Atlantis](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Atlantis_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![BankHeist](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/BankHeist_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![BattleZone](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/BattleZone_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![BeamRider](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/BeamRider_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Berzerk](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Berzerk_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Bowling](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Bowling_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Boxing](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Boxing_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Breakout](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Breakout_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Carnival](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Carnival_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Centipede](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Centipede_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![ChopperCommand](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/ChopperCommand_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![CrazyClimber](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/CrazyClimber_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Defender](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Defender_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![DemonAttack](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/DemonAttack_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![DoubleDunk](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/DoubleDunk_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![ElevatorAction](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/ElevatorAction_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Enduro](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Enduro_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![FishingDerby](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/FishingDerby_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Freeway](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Freeway_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Frostbite](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Frostbite_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Gopher](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Gopher_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Gravitar](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Gravitar_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Hero](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Hero_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![IceHockey](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/IceHockey_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Jamesbond](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Jamesbond_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![JourneyEscape](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/JourneyEscape_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Kangaroo](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Kangaroo_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Krull](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Krull_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![KungFuMaster](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/KungFuMaster_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![MsPacman](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/MsPacman_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![NameThisGame](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/NameThisGame_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Phoenix](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Phoenix_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Pong](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Pong_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Pooyan](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Pooyan_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Qbert](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Qbert_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Riverraid](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Riverraid_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![RoadRunner](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/RoadRunner_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Robotank](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Robotank_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Seaquest](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Seaquest_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Skiing](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Skiing_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Solaris](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Solaris_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![SpaceInvaders](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/SpaceInvaders_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![StarGunner](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/StarGunner_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Surround](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Surround_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Tennis](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Tennis_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![TimePilot](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/TimePilot_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Tutankham](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Tutankham_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![UpNDown](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/UpNDown_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![VideoPinball](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/VideoPinball_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![WizardOfWor](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/WizardOfWor_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![YarsRevenge](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/YarsRevenge_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Zaxxon](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Zaxxon_multi_trial_graph_mean_returns_ma_vs_frames.png) | | |

### PPO Lambda Comparison

<details>
<summary><b>Lambda Comparison Table</b> - click to expand</summary>

Shows scores for all three lambda variants where tested. **Bold** = best score, `-` = not tested.

| Game | ppo_atari (0.95) | ppo_atari_lam85 (0.85) | ppo_atari_lam70 (0.70) |
|------|------------------|------------------------|------------------------|
| ALE/AirRaid-v5 | **8245** | - | - |
| ALE/Alien-v5 | **1453** | 1353 | 1274 |
| ALE/Amidar-v5 | 574 | **580** | - |
| ALE/Assault-v5 | 4059 | **4293** | 3314 |
| ALE/Asterix-v5 | 2967 | **3482** | - |
| ALE/Asteroids-v5 | 1497 | **1554** | - |
| ALE/Atlantis-v5 | **792886** | 754k | 710k |
| ALE/BankHeist-v5 | **1045** | 1045 | - |
| ALE/BattleZone-v5 | 21270 | **26383** | 13857 |
| ALE/BeamRider-v5 | **2765** | - | - |
| ALE/Berzerk-v5 | **1072** | - | - |
| ALE/Bowling-v5 | **46.45** | - | - |
| ALE/Boxing-v5 | **91.17** | - | - |
| ALE/Breakout-v5 | 191 | 292 | **327** |
| ALE/Carnival-v5 | 3071 | 3013 | **3967** |
| ALE/Centipede-v5 | 3917 | - | **4915** |
| ALE/ChopperCommand-v5 | **5355** | - | - |
| ALE/CrazyClimber-v5 | 107183 | **107370** | - |
| ALE/Defender-v5 | 37162 | - | **51439** |
| ALE/DemonAttack-v5 | 7755 | - | **16558** |
| ALE/DoubleDunk-v5 | **-2.38** | - | - |
| ALE/ElevatorAction-v5 | **5446** | 363 | 3933 |
| ALE/Enduro-v5 | 414 | **898** | 872 |
| ALE/FishingDerby-v5 | 22.80 | **27.10** | - |
| ALE/Freeway-v5 | **31.30** | - | - |
| ALE/Frostbite-v5 | **301** | 275 | 267 |
| ALE/Gopher-v5 | 4172 | - | **6508** |
| ALE/Gravitar-v5 | **599** | 253 | 145 |
| ALE/Hero-v5 | 21052 | **28238** | - |
| ALE/IceHockey-v5 | **-3.93** | -5.58 | -7.36 |
| ALE/Jamesbond-v5 | **662** | - | - |
| ALE/JourneyEscape-v5 | -1582 | **-1252** | -1547 |
| ALE/Kangaroo-v5 | 2623 | **9912** | - |
| ALE/Krull-v5 | **7841** | - | - |
| ALE/KungFuMaster-v5 | 18973 | 28334 | **29068** |
| ALE/MsPacman-v5 | 2308 | **2372** | 2297 |
| ALE/NameThisGame-v5 | **5993** | - | - |
| ALE/Phoenix-v5 | 7940 | - | **15659** |
| ALE/Pong-v5 | 15.01 | **16.91** | 12.85 |
| ALE/Pooyan-v5 | 4704 | - | **5716** |
| ALE/Qbert-v5 | **15094** | - | - |
| ALE/Riverraid-v5 | 7319 | **9428** | - |
| ALE/RoadRunner-v5 | 24204 | **37015** | - |
| ALE/Robotank-v5 | **20.07** | 8.24 | 2.59 |
| ALE/Seaquest-v5 | **1796** | - | - |
| ALE/Skiing-v5 | **-19340** | -22980 | -29975 |
| ALE/Solaris-v5 | **2094** | - | - |
| ALE/SpaceInvaders-v5 | **726** | - | - |
| ALE/StarGunner-v5 | 31862 | - | **47495** |
| ALE/Surround-v5 | **-2.52** | - | -6.79 |
| ALE/Tennis-v5 | -7.66 | **-4.41** | - |
| ALE/TimePilot-v5 | **4668** | - | - |
| ALE/Tutankham-v5 | 203 | **217** | - |
| ALE/UpNDown-v5 | **182472** | - | - |
| ALE/VideoPinball-v5 | 31385 | - | **56746** |
| ALE/WizardOfWor-v5 | **5814** | 5466 | 4740 |
| ALE/YarsRevenge-v5 | **17120** | - | - |
| ALE/Zaxxon-v5 | **10756** | - | - |

</details>

### Running Atari Benchmarks

All games use the same spec file with variable substitution for the environment:

```bash
# A2C
source .env && slm-lab run-remote --gpu -s env=ALE/Breakout-v5 -s max_frame=1e7 \
  slm_lab/spec/benchmark/a2c/a2c_gae_atari.json a2c_gae_atari train -n breakout-a2c

# PPO
source .env && slm-lab run-remote --gpu -s env=ALE/Breakout-v5 -s max_frame=1e7 \
  slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam70 train -n breakout-ppo
```

### Download and Replay

```bash
# List Atari experiments (requires HF_REPO=SLM-Lab/benchmark in .env)
source .env && slm-lab list | grep atari

# Download a specific game
source .env && slm-lab pull ppo_atari_breakout

# Replay
slm-lab run slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam70 enjoy@data/ppo_atari_lam70_breakout_*/ppo_atari_lam70_breakout_t0_spec.json
```

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
