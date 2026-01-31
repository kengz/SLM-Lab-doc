# Atari Environment Benchmark

## PPO Atari Results (v5)

SLM Lab v5 validates PPO on Gymnasium ALE environments. **54 games tested** with all results available on [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark).

For the complete methodology and full results table, see [docs/BENCHMARKS.md](https://github.com/kengz/SLM-Lab/blob/master/docs/BENCHMARKS.md) in the code repository.

{% hint style="info" %}
**v5 Environment Changes:** Gymnasium ALE v5 uses sticky actions (`repeat_action_probability=0.25`) per [Machado et al. (2018)](https://arxiv.org/abs/1709.06009) best practices. This makes environments harder than the older NoFrameskip-v4 variants.
{% endhint %}

### Configuration

* **Specs:** [ppo_atari.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/ppo/ppo_atari.json)
* **Training:** 10M frames, 16 parallel envs, ConvNet [32,64,64]+512fc (Nature CNN)
* **Key settings:** `life_loss_info=true`, `clip_vloss=true`, AdamW (lr=2.5e-4), minibatch=256

### Lambda Variants

Different games benefit from different lambda values for GAE. All variants use the same spec file:

| SPEC_NAME | Lambda | Best for |
|-----------|--------|----------|
| ppo_atari | 0.95 | Strategic games (default) |
| ppo_atari_lam85 | 0.85 | Mixed games |
| ppo_atari_lam70 | 0.70 | Action games |

### Selected v5 Results

| Game | Score | Lambda | HuggingFace |
|------|-------|--------|-------------|
| ALE/Breakout-v5 | 327 | lam70 | [ppo_atari_lam70_breakout_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_breakout_2026_01_07_110559) |
| ALE/Pong-v5 | 16.9 | lam85 | [ppo_atari_lam85_pong_2026_01_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_pong_2026_01_08_094454) |
| ALE/Qbert-v5 | 15094 | lam95 | [ppo_atari_qbert_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_qbert_2026_01_06_111801) |
| ALE/BeamRider-v5 | 2765 | lam95 | [ppo_atari_beamrider_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_beamrider_2026_01_06_112533) |
| ALE/SpaceInvaders-v5 | 726 | lam95 | [ppo_atari_spaceinvaders_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_spaceinvaders_2026_01_07_102346) |
| ALE/Seaquest-v5 | 1796 | lam95 | [ppo_atari_seaquest_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_seaquest_2026_01_06_183440) |
| ALE/KungFuMaster-v5 | 29068 | lam70 | [ppo_atari_lam70_kungfumaster_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_kungfumaster_2026_01_07_111317) |
| ALE/MsPacman-v5 | 2372 | lam85 | [ppo_atari_lam85_mspacman_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_mspacman_2026_01_07_223522) |
| ALE/Atlantis-v5 | 792886 | lam95 | [ppo_atari_atlantis_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_atlantis_2026_01_06_120440) |
| ALE/Enduro-v5 | 898 | lam85 | [ppo_atari_lam85_enduro_2026_01_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_enduro_2026_01_08_095448) |

**Skipped** (hard exploration): Adventure, MontezumaRevenge, Pitfall, PrivateEye, Venture

### Full Game Table

See [docs/BENCHMARKS.md](https://github.com/kengz/SLM-Lab/blob/master/docs/BENCHMARKS.md) for the complete table with all 54 games and direct HuggingFace links.

<details>
<summary><b>All 54 Games</b> - click to expand</summary>

| Game | Score | SPEC_NAME |
|------|-------|-----------|
| ALE/AirRaid-v5 | 8245 | ppo_atari |
| ALE/Alien-v5 | 1453 | ppo_atari |
| ALE/Amidar-v5 | 580 | ppo_atari_lam85 |
| ALE/Assault-v5 | 4293 | ppo_atari_lam85 |
| ALE/Asterix-v5 | 3482 | ppo_atari_lam85 |
| ALE/Asteroids-v5 | 1554 | ppo_atari_lam85 |
| ALE/Atlantis-v5 | 792886 | ppo_atari |
| ALE/BankHeist-v5 | 1045 | ppo_atari |
| ALE/BattleZone-v5 | 26383 | ppo_atari_lam85 |
| ALE/BeamRider-v5 | 2765 | ppo_atari |
| ALE/Berzerk-v5 | 1072 | ppo_atari |
| ALE/Bowling-v5 | 46.45 | ppo_atari |
| ALE/Boxing-v5 | 91.17 | ppo_atari |
| ALE/Breakout-v5 | 327 | ppo_atari_lam70 |
| ALE/Carnival-v5 | 3967 | ppo_atari_lam70 |
| ALE/Centipede-v5 | 4915 | ppo_atari_lam70 |
| ALE/ChopperCommand-v5 | 5355 | ppo_atari |
| ALE/CrazyClimber-v5 | 107370 | ppo_atari_lam85 |
| ALE/Defender-v5 | 51439 | ppo_atari_lam70 |
| ALE/DemonAttack-v5 | 16558 | ppo_atari_lam70 |
| ALE/DoubleDunk-v5 | -2.38 | ppo_atari |
| ALE/ElevatorAction-v5 | 5446 | ppo_atari |
| ALE/Enduro-v5 | 898 | ppo_atari_lam85 |
| ALE/FishingDerby-v5 | 27.10 | ppo_atari_lam85 |
| ALE/Freeway-v5 | 31.30 | ppo_atari |
| ALE/Frostbite-v5 | 301 | ppo_atari |
| ALE/Gopher-v5 | 6508 | ppo_atari_lam70 |
| ALE/Gravitar-v5 | 599 | ppo_atari |
| ALE/Hero-v5 | 28238 | ppo_atari_lam85 |
| ALE/IceHockey-v5 | -3.93 | ppo_atari |
| ALE/Jamesbond-v5 | 662 | ppo_atari |
| ALE/JourneyEscape-v5 | -1252 | ppo_atari_lam85 |
| ALE/Kangaroo-v5 | 9912 | ppo_atari_lam85 |
| ALE/Krull-v5 | 7841 | ppo_atari |
| ALE/KungFuMaster-v5 | 29068 | ppo_atari_lam70 |
| ALE/MsPacman-v5 | 2372 | ppo_atari_lam85 |
| ALE/NameThisGame-v5 | 5993 | ppo_atari |
| ALE/Phoenix-v5 | 15659 | ppo_atari_lam70 |
| ALE/Pong-v5 | 16.91 | ppo_atari_lam85 |
| ALE/Pooyan-v5 | 5716 | ppo_atari_lam70 |
| ALE/Qbert-v5 | 15094 | ppo_atari |
| ALE/Riverraid-v5 | 9428 | ppo_atari_lam85 |
| ALE/RoadRunner-v5 | 37015 | ppo_atari_lam85 |
| ALE/Robotank-v5 | 20.07 | ppo_atari |
| ALE/Seaquest-v5 | 1796 | ppo_atari |
| ALE/Skiing-v5 | -19340 | ppo_atari |
| ALE/Solaris-v5 | 2094 | ppo_atari |
| ALE/SpaceInvaders-v5 | 726 | ppo_atari |
| ALE/StarGunner-v5 | 47495 | ppo_atari_lam70 |
| ALE/Surround-v5 | -2.52 | ppo_atari |
| ALE/Tennis-v5 | -4.41 | ppo_atari_lam85 |
| ALE/TimePilot-v5 | 4668 | ppo_atari |
| ALE/Tutankham-v5 | 217 | ppo_atari_lam85 |
| ALE/UpNDown-v5 | 182472 | ppo_atari |
| ALE/VideoPinball-v5 | 56746 | ppo_atari_lam70 |
| ALE/WizardOfWor-v5 | 5814 | ppo_atari |
| ALE/YarsRevenge-v5 | 17120 | ppo_atari |
| ALE/Zaxxon-v5 | 10756 | ppo_atari |

</details>

### Running Atari Benchmarks

```bash
# Using template spec with variable substitution
slm-lab run -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam70 train
slm-lab run -s env=ALE/Qbert-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train
slm-lab run -s env=ALE/Pong-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari_lam85 train
```

### Download and Replay

```bash
# List Atari experiments
slm-lab list | grep atari

# Download a specific game
slm-lab pull ppo_atari_breakout

# Replay
slm-lab run _ _ enjoy@data/ppo_atari_*/ppo_atari_t0_spec.json
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
