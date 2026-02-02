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

### Running Benchmarks

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

---

## Results

| ENV | Score | SPEC_NAME | HF Repo |
|-----|-------|-----------|---------|
| ALE/AirRaid-v5 | 5067 | a2c_gae_atari | [a2c_gae_atari_airraid_2026_02_01_082446](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_airraid_2026_02_01_082446) |
| | 8245 | ppo_atari | [ppo_atari_airraid_2026_01_06_113119](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_airraid_2026_01_06_113119) |
| ALE/Alien-v5 | 1488 | a2c_gae_atari | [a2c_gae_atari_alien_2026_02_01_000858](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_alien_2026_02_01_000858) |
| | 1453 | ppo_atari | [ppo_atari_alien_2026_01_06_112514](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_alien_2026_01_06_112514) |
| ALE/Amidar-v5 | 330 | a2c_gae_atari | [a2c_gae_atari_amidar_2026_02_01_082251](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_amidar_2026_02_01_082251) |
| | 580 | ppo_atari_lam85 | [ppo_atari_lam85_amidar_2026_01_07_223416](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_amidar_2026_01_07_223416) |
| ALE/Assault-v5 | 1646 | a2c_gae_atari | [a2c_gae_atari_assault_2026_02_01_082252](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_assault_2026_02_01_082252) |
| | 4293 | ppo_atari_lam85 | [ppo_atari_lam85_assault_2026_01_08_130044](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_assault_2026_01_08_130044) |
| ALE/Asterix-v5 | 2712 | a2c_gae_atari | [a2c_gae_atari_asterix_2026_02_01_082315](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_asterix_2026_02_01_082315) |
| | 3482 | ppo_atari_lam85 | [ppo_atari_lam85_asterix_2026_01_07_223445](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_asterix_2026_01_07_223445) |
| ALE/Asteroids-v5 | 2106 | a2c_gae_atari | [a2c_gae_atari_asteroids_2026_02_01_082328](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_asteroids_2026_02_01_082328) |
| | 1554 | ppo_atari_lam85 | [ppo_atari_lam85_asteroids_2026_01_07_224245](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_asteroids_2026_01_07_224245) |
| ALE/Atlantis-v5 | 873365 | a2c_gae_atari | [a2c_gae_atari_atlantis_2026_02_01_082330](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_atlantis_2026_02_01_082330) |
| | 792886 | ppo_atari | [ppo_atari_atlantis_2026_01_06_120440](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_atlantis_2026_01_06_120440) |
| ALE/BankHeist-v5 | 1099 | a2c_gae_atari | [a2c_gae_atari_bankheist_2026_02_01_082403](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_bankheist_2026_02_01_082403) |
| | 1045 | ppo_atari | [ppo_atari_bankheist_2026_01_06_121042](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_bankheist_2026_01_06_121042) |
| ALE/BattleZone-v5 | 2437 | a2c_gae_atari | [a2c_gae_atari_battlezone_2026_02_01_082425](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_battlezone_2026_02_01_082425) |
| | 26383 | ppo_atari_lam85 | [ppo_atari_lam85_battlezone_2026_01_08_094729](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_battlezone_2026_01_08_094729) |
| ALE/BeamRider-v5 | 2767 | a2c_gae_atari | [a2c_gae_atari_beamrider_2026_02_01_000921](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_beamrider_2026_02_01_000921) |
| | 2765 | ppo_atari | [ppo_atari_beamrider_2026_01_06_112533](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_beamrider_2026_01_06_112533) |
| ALE/Berzerk-v5 | 439 | a2c_gae_atari | [a2c_gae_atari_berzerk_2026_02_01_082540](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_berzerk_2026_02_01_082540) |
| | 1072 | ppo_atari | [ppo_atari_berzerk_2026_01_06_112515](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_berzerk_2026_01_06_112515) |
| ALE/Bowling-v5 | 23.96 | a2c_gae_atari | [a2c_gae_atari_bowling_2026_02_01_082529](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_bowling_2026_02_01_082529) |
| | 46.45 | ppo_atari | [ppo_atari_bowling_2026_01_06_113148](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_bowling_2026_01_06_113148) |
| ALE/Boxing-v5 | 1.80 | a2c_gae_atari | [a2c_gae_atari_boxing_2026_02_01_082539](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_boxing_2026_02_01_082539) |
| | 91.17 | ppo_atari | [ppo_atari_boxing_2026_01_06_112531](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_boxing_2026_01_06_112531) |
| ALE/Breakout-v5 | 273 | a2c_gae_atari | [a2c_gae_atari_breakout_2026_01_31_213610](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_breakout_2026_01_31_213610) |
| | 327 | ppo_atari_lam70 | [ppo_atari_lam70_breakout_2026_01_07_110559](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_breakout_2026_01_07_110559) |
| ALE/Carnival-v5 | 2170 | a2c_gae_atari | [a2c_gae_atari_carnival_2026_02_01_082726](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_carnival_2026_02_01_082726) |
| | 3967 | ppo_atari_lam70 | [ppo_atari_lam70_carnival_2026_01_07_144738](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_carnival_2026_01_07_144738) |
| ALE/Centipede-v5 | 1382 | a2c_gae_atari | [a2c_gae_atari_centipede_2026_02_01_082643](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_centipede_2026_02_01_082643) |
| | 4915 | ppo_atari_lam70 | [ppo_atari_lam70_centipede_2026_01_07_223557](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_centipede_2026_01_07_223557) |
| ALE/ChopperCommand-v5 | 2446 | a2c_gae_atari | [a2c_gae_atari_choppercommand_2026_02_01_082626](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_choppercommand_2026_02_01_082626) |
| | 5355 | ppo_atari | [ppo_atari_choppercommand_2026_01_07_110539](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_choppercommand_2026_01_07_110539) |
| ALE/CrazyClimber-v5 | 96943 | a2c_gae_atari | [a2c_gae_atari_crazyclimber_2026_02_01_082625](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_crazyclimber_2026_02_01_082625) |
| | 107370 | ppo_atari_lam85 | [ppo_atari_lam85_crazyclimber_2026_01_07_223609](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_crazyclimber_2026_01_07_223609) |
| ALE/Defender-v5 | 33149 | a2c_gae_atari | [a2c_gae_atari_defender_2026_02_01_082658](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_defender_2026_02_01_082658) |
| | 51439 | ppo_atari_lam70 | [ppo_atari_lam70_defender_2026_01_07_205238](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_defender_2026_01_07_205238) |
| ALE/DemonAttack-v5 | 2962 | a2c_gae_atari | [a2c_gae_atari_demonattack_2026_02_01_082717](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_demonattack_2026_02_01_082717) |
| | 16558 | ppo_atari_lam70 | [ppo_atari_lam70_demonattack_2026_01_07_111315](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_demonattack_2026_01_07_111315) |
| ALE/DoubleDunk-v5 | -1.69 | a2c_gae_atari | [a2c_gae_atari_doubledunk_2026_02_01_082901](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_doubledunk_2026_02_01_082901) |
| | -2.38 | ppo_atari | [ppo_atari_doubledunk_2026_01_07_110802](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_doubledunk_2026_01_07_110802) |
| ALE/ElevatorAction-v5 | 731 | a2c_gae_atari | [a2c_gae_atari_elevatoraction_2026_02_01_082908](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_elevatoraction_2026_02_01_082908) |
| | 5446 | ppo_atari | [ppo_atari_elevatoraction_2026_01_06_113129](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_elevatoraction_2026_01_06_113129) |
| ALE/Enduro-v5 | 681 | a2c_gae_atari | [a2c_gae_atari_enduro_2026_02_01_001123](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_enduro_2026_02_01_001123) |
| | 898 | ppo_atari_lam85 | [ppo_atari_lam85_enduro_2026_01_08_095448](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_enduro_2026_01_08_095448) |
| ALE/FishingDerby-v5 | -16.38 | a2c_gae_atari | [a2c_gae_atari_fishingderby_2026_02_01_082906](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_fishingderby_2026_02_01_082906) |
| | 27.10 | ppo_atari_lam85 | [ppo_atari_lam85_fishingderby_2026_01_08_094158](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_fishingderby_2026_01_08_094158) |
| ALE/Freeway-v5 | 23.13 | a2c_gae_atari | [a2c_gae_atari_freeway_2026_02_01_082931](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_freeway_2026_02_01_082931) |
| | 31.30 | ppo_atari | [ppo_atari_freeway_2026_01_06_182318](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_freeway_2026_01_06_182318) |
| ALE/Frostbite-v5 | 266 | a2c_gae_atari | [a2c_gae_atari_frostbite_2026_02_01_082915](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_frostbite_2026_02_01_082915) |
| | 301 | ppo_atari | [ppo_atari_frostbite_2026_01_06_112556](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_frostbite_2026_01_06_112556) |
| ALE/Gopher-v5 | 984 | a2c_gae_atari | [a2c_gae_atari_gopher_2026_02_01_133323](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_gopher_2026_02_01_133323) |
| | 6508 | ppo_atari_lam70 | [ppo_atari_lam70_gopher_2026_01_07_170451](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_gopher_2026_01_07_170451) |
| ALE/Gravitar-v5 | 270 | a2c_gae_atari | [a2c_gae_atari_gravitar_2026_02_01_133244](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_gravitar_2026_02_01_133244) |
| | 599 | ppo_atari | [ppo_atari_gravitar_2026_01_06_112548](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_gravitar_2026_01_06_112548) |
| ALE/Hero-v5 | 18680 | a2c_gae_atari | [a2c_gae_atari_hero_2026_02_01_175903](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_hero_2026_02_01_175903) |
| | 28238 | ppo_atari_lam85 | [ppo_atari_lam85_hero_2026_01_07_223619](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_hero_2026_01_07_223619) |
| ALE/IceHockey-v5 | -5.92 | a2c_gae_atari | [a2c_gae_atari_icehockey_2026_02_01_175745](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_icehockey_2026_02_01_175745) |
| | -3.93 | ppo_atari | [ppo_atari_icehockey_2026_01_06_183721](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_icehockey_2026_01_06_183721) |
| ALE/Jamesbond-v5 | 460 | a2c_gae_atari | [a2c_gae_atari_jamesbond_2026_02_01_175945](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_jamesbond_2026_02_01_175945) |
| | 662 | ppo_atari | [ppo_atari_jamesbond_2026_01_06_183717](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_jamesbond_2026_01_06_183717) |
| ALE/JourneyEscape-v5 | -965 | a2c_gae_atari | [a2c_gae_atari_journeyescape_2026_02_01_084415](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_journeyescape_2026_02_01_084415) |
| | -1252 | ppo_atari_lam85 | [ppo_atari_lam85_journeyescape_2026_01_08_094842](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_journeyescape_2026_01_08_094842) |
| ALE/Kangaroo-v5 | 322 | a2c_gae_atari | [a2c_gae_atari_kangaroo_2026_02_01_084415](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_kangaroo_2026_02_01_084415) |
| | 9912 | ppo_atari_lam85 | [ppo_atari_lam85_kangaroo_2026_01_07_110838](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_kangaroo_2026_01_07_110838) |
| ALE/Krull-v5 | 7519 | a2c_gae_atari | [a2c_gae_atari_krull_2026_02_01_084420](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_krull_2026_02_01_084420) |
| | 7841 | ppo_atari | [ppo_atari_krull_2026_01_07_110747](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_krull_2026_01_07_110747) |
| ALE/KungFuMaster-v5 | 23006 | a2c_gae_atari | [a2c_gae_atari_kungfumaster_2026_02_01_085101](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_kungfumaster_2026_02_01_085101) |
| | 29068 | ppo_atari_lam70 | [ppo_atari_lam70_kungfumaster_2026_01_07_111317](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_kungfumaster_2026_01_07_111317) |
| ALE/MsPacman-v5 | 2110 | a2c_gae_atari | [a2c_gae_atari_mspacman_2026_02_01_001100](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_mspacman_2026_02_01_001100) |
| | 2372 | ppo_atari_lam85 | [ppo_atari_lam85_mspacman_2026_01_07_223522](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_mspacman_2026_01_07_223522) |
| ALE/NameThisGame-v5 | 5412 | a2c_gae_atari | [a2c_gae_atari_namethisgame_2026_02_01_132733](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_namethisgame_2026_02_01_132733) |
| | 5993 | ppo_atari | [ppo_atari_namethisgame_2026_01_06_182952](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_namethisgame_2026_01_06_182952) |
| ALE/Phoenix-v5 | 5635 | a2c_gae_atari | [a2c_gae_atari_phoenix_2026_02_01_085101](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_phoenix_2026_02_01_085101) |
| | 15659 | ppo_atari_lam70 | [ppo_atari_lam70_phoenix_2026_01_07_110832](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_phoenix_2026_01_07_110832) |
| ALE/Pong-v5 | 10.17 | a2c_gae_atari | [a2c_gae_atari_pong_2026_01_31_213635](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_pong_2026_01_31_213635) |
| | 16.91 | ppo_atari_lam85 | [ppo_atari_lam85_pong_2026_01_08_094454](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_pong_2026_01_08_094454) |
| ALE/Pooyan-v5 | 2997 | a2c_gae_atari | [a2c_gae_atari_pooyan_2026_02_01_132748](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_pooyan_2026_02_01_132748) |
| | 5716 | ppo_atari_lam70 | [ppo_atari_lam70_pooyan_2026_01_07_224346](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_pooyan_2026_01_07_224346) |
| ALE/Qbert-v5 | 12619 | a2c_gae_atari | [a2c_gae_atari_qbert_2026_01_31_213720](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_qbert_2026_01_31_213720) |
| | 15094 | ppo_atari | [ppo_atari_qbert_2026_01_06_111801](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_qbert_2026_01_06_111801) |
| ALE/Riverraid-v5 | 6558 | a2c_gae_atari | [a2c_gae_atari_riverraid_2026_02_01_132507](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_riverraid_2026_02_01_132507) |
| | 9428 | ppo_atari_lam85 | [ppo_atari_lam85_riverraid_2026_01_07_204356](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_riverraid_2026_01_07_204356) |
| ALE/RoadRunner-v5 | 29810 | a2c_gae_atari | [a2c_gae_atari_roadrunner_2026_02_01_132509](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_roadrunner_2026_02_01_132509) |
| | 37015 | ppo_atari_lam85 | [ppo_atari_lam85_roadrunner_2026_01_07_145913](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_roadrunner_2026_01_07_145913) |
| ALE/Robotank-v5 | 2.80 | a2c_gae_atari | [a2c_gae_atari_robotank_2026_02_01_132434](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_robotank_2026_02_01_132434) |
| | 20.07 | ppo_atari | [ppo_atari_robotank_2026_01_06_183413](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_robotank_2026_01_06_183413) |
| ALE/Seaquest-v5 | 850 | a2c_gae_atari | [a2c_gae_atari_seaquest_2026_02_01_001001](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_seaquest_2026_02_01_001001) |
| | 1796 | ppo_atari | [ppo_atari_seaquest_2026_01_06_183440](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_seaquest_2026_01_06_183440) |
| ALE/Skiing-v5 | -14235 | a2c_gae_atari | [a2c_gae_atari_skiing_2026_02_01_132451](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_skiing_2026_02_01_132451) |
| | -19340 | ppo_atari | [ppo_atari_skiing_2026_01_06_183424](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_skiing_2026_01_06_183424) |
| ALE/Solaris-v5 | 2224 | a2c_gae_atari | [a2c_gae_atari_solaris_2026_02_01_212137](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_solaris_2026_02_01_212137) |
| | 2094 | ppo_atari | [ppo_atari_solaris_2026_01_06_192643](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_solaris_2026_01_06_192643) |
| ALE/SpaceInvaders-v5 | 784 | a2c_gae_atari | [a2c_gae_atari_spaceinvaders_2026_02_01_000950](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_spaceinvaders_2026_02_01_000950) |
| | 726 | ppo_atari | [ppo_atari_spaceinvaders_2026_01_07_102346](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_spaceinvaders_2026_01_07_102346) |
| ALE/StarGunner-v5 | 8665 | a2c_gae_atari | [a2c_gae_atari_stargunner_2026_02_01_132406](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_stargunner_2026_02_01_132406) |
| | 47495 | ppo_atari_lam70 | [ppo_atari_lam70_stargunner_2026_01_07_111404](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_stargunner_2026_01_07_111404) |
| ALE/Surround-v5 | -9.72 | a2c_gae_atari | [a2c_gae_atari_surround_2026_02_01_132215](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_surround_2026_02_01_132215) |
| | -2.52 | ppo_atari | [ppo_atari_surround_2026_01_07_102404](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_surround_2026_01_07_102404) |
| ALE/Tennis-v5 | -2873 | a2c_gae_atari | [a2c_gae_atari_tennis_2026_02_01_175829](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_tennis_2026_02_01_175829) |
| | -4.41 | ppo_atari_lam85 | [ppo_atari_lam85_tennis_2026_01_07_223532](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_tennis_2026_01_07_223532) |
| ALE/TimePilot-v5 | 3376 | a2c_gae_atari | [a2c_gae_atari_timepilot_2026_02_01_175930](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_timepilot_2026_02_01_175930) |
| | 4668 | ppo_atari | [ppo_atari_timepilot_2026_01_07_101010](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_timepilot_2026_01_07_101010) |
| ALE/Tutankham-v5 | 167 | a2c_gae_atari | [a2c_gae_atari_tutankham_2026_02_01_132347](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_tutankham_2026_02_01_132347) |
| | 217 | ppo_atari_lam85 | [ppo_atari_lam85_tutankham_2026_01_08_095251](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_tutankham_2026_01_08_095251) |
| ALE/UpNDown-v5 | 57099 | a2c_gae_atari | [a2c_gae_atari_upndown_2026_02_01_132435](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_upndown_2026_02_01_132435) |
| | 182472 | ppo_atari | [ppo_atari_upndown_2026_01_07_105708](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_upndown_2026_01_07_105708) |
| ALE/VideoPinball-v5 | 25310 | a2c_gae_atari | [a2c_gae_atari_videopinball_2026_02_01_083457](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_videopinball_2026_02_01_083457) |
| | 56746 | ppo_atari_lam70 | [ppo_atari_lam70_videopinball_2026_01_07_224359](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_videopinball_2026_01_07_224359) |
| ALE/WizardOfWor-v5 | 2682 | a2c_gae_atari | [a2c_gae_atari_wizardofwor_2026_02_01_132449](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_wizardofwor_2026_02_01_132449) |
| | 5814 | ppo_atari | [ppo_atari_wizardofwor_2026_01_06_221154](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_wizardofwor_2026_01_06_221154) |
| ALE/YarsRevenge-v5 | 24371 | a2c_gae_atari | [a2c_gae_atari_yarsrevenge_2026_02_01_132224](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_yarsrevenge_2026_02_01_132224) |
| | 17120 | ppo_atari | [ppo_atari_yarsrevenge_2026_01_06_221154](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_yarsrevenge_2026_01_06_221154) |
| ALE/Zaxxon-v5 | 29.46 | a2c_gae_atari | [a2c_gae_atari_zaxxon_2026_02_01_131758](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_zaxxon_2026_02_01_131758) |
| | 10756 | ppo_atari | [ppo_atari_zaxxon_2026_01_06_221154](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_zaxxon_2026_01_06_221154) |

**Skipped** (hard exploration): Adventure, MontezumaRevenge, Pitfall, PrivateEye, Venture

### Training Curves

Multi-trial comparison plots showing A2C vs PPO mean returns (moving average) vs training frames. Shaded regions show standard deviation across 4 sessions.

| | | |
|:---:|:---:|:---:|
| ![AirRaid](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/AirRaid_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Alien](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Alien_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Amidar](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Amidar_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![Assault](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Assault_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Asterix](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Asterix_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Asteroids](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Asteroids_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![Atlantis](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Atlantis_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![BankHeist](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/BankHeist_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![BattleZone](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/BattleZone_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![BeamRider](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/BeamRider_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Berzerk](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Berzerk_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Bowling](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Bowling_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![Boxing](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Boxing_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Breakout](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Breakout_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Carnival](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Carnival_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![Centipede](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Centipede_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![ChopperCommand](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/ChopperCommand_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![CrazyClimber](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/CrazyClimber_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![Defender](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Defender_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![DemonAttack](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/DemonAttack_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![DoubleDunk](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/DoubleDunk_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![ElevatorAction](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/ElevatorAction_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Enduro](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Enduro_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![FishingDerby](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/FishingDerby_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![Freeway](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Freeway_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Frostbite](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Frostbite_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Gopher](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Gopher_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![Gravitar](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Gravitar_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Hero](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Hero_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![IceHockey](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/IceHockey_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![Jamesbond](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Jamesbond_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![JourneyEscape](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/JourneyEscape_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Kangaroo](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Kangaroo_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![Krull](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Krull_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![KungFuMaster](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/KungFuMaster_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![MsPacman](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/MsPacman_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![NameThisGame](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/NameThisGame_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Phoenix](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Phoenix_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Pong](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Pong_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![Pooyan](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Pooyan_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Qbert](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Qbert_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Riverraid](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Riverraid_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![RoadRunner](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/RoadRunner_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Robotank](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Robotank_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Seaquest](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Seaquest_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![Skiing](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Skiing_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Solaris](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Solaris_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![SpaceInvaders](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/SpaceInvaders_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![StarGunner](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/StarGunner_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Surround](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Surround_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Tennis](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Tennis_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![TimePilot](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/TimePilot_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![Tutankham](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Tutankham_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![UpNDown](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/UpNDown_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![VideoPinball](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/VideoPinball_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![WizardOfWor](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/WizardOfWor_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | ![YarsRevenge](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/YarsRevenge_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) |
| ![Zaxxon](https://huggingface.co/datasets/SLM-Lab/benchmark/resolve/main/docs/plots/Zaxxon_multi_trial_graph_mean_returns_ma_vs_frames.png?v=20260202) | | |

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
