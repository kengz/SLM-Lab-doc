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
<summary><b>All 54 Games with HuggingFace Links</b> - click to expand</summary>

| Game | Score | SPEC_NAME | HuggingFace |
|------|-------|-----------|-------------|
| ALE/AirRaid-v5 | 8245 | ppo_atari | [ppo_atari_airraid_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_airraid_2026_01_06_113119) |
| ALE/Alien-v5 | 1453 | ppo_atari | [ppo_atari_alien_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_alien_2026_01_06_112514) |
| ALE/Amidar-v5 | 580 | ppo_atari_lam85 | [ppo_atari_lam85_amidar_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_amidar_2026_01_07_223416) |
| ALE/Assault-v5 | 4293 | ppo_atari_lam85 | [ppo_atari_lam85_assault_2026_01_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_assault_2026_01_08_130044) |
| ALE/Asterix-v5 | 3482 | ppo_atari_lam85 | [ppo_atari_lam85_asterix_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_asterix_2026_01_07_223445) |
| ALE/Asteroids-v5 | 1554 | ppo_atari_lam85 | [ppo_atari_lam85_asteroids_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_asteroids_2026_01_07_224245) |
| ALE/Atlantis-v5 | 792886 | ppo_atari | [ppo_atari_atlantis_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_atlantis_2026_01_06_120440) |
| ALE/BankHeist-v5 | 1045 | ppo_atari | [ppo_atari_bankheist_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_bankheist_2026_01_06_121042) |
| ALE/BattleZone-v5 | 26383 | ppo_atari_lam85 | [ppo_atari_lam85_battlezone_2026_01_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_battlezone_2026_01_08_094729) |
| ALE/BeamRider-v5 | 2765 | ppo_atari | [ppo_atari_beamrider_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_beamrider_2026_01_06_112533) |
| ALE/Berzerk-v5 | 1072 | ppo_atari | [ppo_atari_berzerk_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_berzerk_2026_01_06_112515) |
| ALE/Bowling-v5 | 46.45 | ppo_atari | [ppo_atari_bowling_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_bowling_2026_01_06_113148) |
| ALE/Boxing-v5 | 91.17 | ppo_atari | [ppo_atari_boxing_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_boxing_2026_01_06_112531) |
| ALE/Breakout-v5 | 327 | ppo_atari_lam70 | [ppo_atari_lam70_breakout_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_breakout_2026_01_07_110559) |
| ALE/Carnival-v5 | 3967 | ppo_atari_lam70 | [ppo_atari_lam70_carnival_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_carnival_2026_01_07_144738) |
| ALE/Centipede-v5 | 4915 | ppo_atari_lam70 | [ppo_atari_lam70_centipede_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_centipede_2026_01_07_223557) |
| ALE/ChopperCommand-v5 | 5355 | ppo_atari | [ppo_atari_choppercommand_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_choppercommand_2026_01_07_110539) |
| ALE/CrazyClimber-v5 | 107370 | ppo_atari_lam85 | [ppo_atari_lam85_crazyclimber_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_crazyclimber_2026_01_07_223609) |
| ALE/Defender-v5 | 51439 | ppo_atari_lam70 | [ppo_atari_lam70_defender_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_defender_2026_01_07_205238) |
| ALE/DemonAttack-v5 | 16558 | ppo_atari_lam70 | [ppo_atari_lam70_demonattack_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_demonattack_2026_01_07_111315) |
| ALE/DoubleDunk-v5 | -2.38 | ppo_atari | [ppo_atari_doubledunk_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_doubledunk_2026_01_07_110802) |
| ALE/ElevatorAction-v5 | 5446 | ppo_atari | [ppo_atari_elevatoraction_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_elevatoraction_2026_01_06_113129) |
| ALE/Enduro-v5 | 898 | ppo_atari_lam85 | [ppo_atari_lam85_enduro_2026_01_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_enduro_2026_01_08_095448) |
| ALE/FishingDerby-v5 | 27.10 | ppo_atari_lam85 | [ppo_atari_lam85_fishingderby_2026_01_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_fishingderby_2026_01_08_094158) |
| ALE/Freeway-v5 | 31.30 | ppo_atari | [ppo_atari_freeway_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_freeway_2026_01_06_182318) |
| ALE/Frostbite-v5 | 301 | ppo_atari | [ppo_atari_frostbite_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_frostbite_2026_01_06_112556) |
| ALE/Gopher-v5 | 6508 | ppo_atari_lam70 | [ppo_atari_lam70_gopher_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_gopher_2026_01_07_170451) |
| ALE/Gravitar-v5 | 599 | ppo_atari | [ppo_atari_gravitar_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_gravitar_2026_01_06_112548) |
| ALE/Hero-v5 | 28238 | ppo_atari_lam85 | [ppo_atari_lam85_hero_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_hero_2026_01_07_223619) |
| ALE/IceHockey-v5 | -3.93 | ppo_atari | [ppo_atari_icehockey_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_icehockey_2026_01_06_183721) |
| ALE/Jamesbond-v5 | 662 | ppo_atari | [ppo_atari_jamesbond_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_jamesbond_2026_01_06_183717) |
| ALE/JourneyEscape-v5 | -1252 | ppo_atari_lam85 | [ppo_atari_lam85_journeyescape_2026_01_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_journeyescape_2026_01_08_094842) |
| ALE/Kangaroo-v5 | 9912 | ppo_atari_lam85 | [ppo_atari_lam85_kangaroo_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_kangaroo_2026_01_07_110838) |
| ALE/Krull-v5 | 7841 | ppo_atari | [ppo_atari_krull_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_krull_2026_01_07_110747) |
| ALE/KungFuMaster-v5 | 29068 | ppo_atari_lam70 | [ppo_atari_lam70_kungfumaster_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_kungfumaster_2026_01_07_111317) |
| ALE/MsPacman-v5 | 2372 | ppo_atari_lam85 | [ppo_atari_lam85_mspacman_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_mspacman_2026_01_07_223522) |
| ALE/NameThisGame-v5 | 5993 | ppo_atari | [ppo_atari_namethisgame_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_namethisgame_2026_01_06_182952) |
| ALE/Phoenix-v5 | 15659 | ppo_atari_lam70 | [ppo_atari_lam70_phoenix_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_phoenix_2026_01_07_110832) |
| ALE/Pong-v5 | 16.91 | ppo_atari_lam85 | [ppo_atari_lam85_pong_2026_01_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_pong_2026_01_08_094454) |
| ALE/Pooyan-v5 | 5716 | ppo_atari_lam70 | [ppo_atari_lam70_pooyan_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_pooyan_2026_01_07_224346) |
| ALE/Qbert-v5 | 15094 | ppo_atari | [ppo_atari_qbert_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_qbert_2026_01_06_111801) |
| ALE/Riverraid-v5 | 9428 | ppo_atari_lam85 | [ppo_atari_lam85_riverraid_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_riverraid_2026_01_07_204356) |
| ALE/RoadRunner-v5 | 37015 | ppo_atari_lam85 | [ppo_atari_lam85_roadrunner_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_roadrunner_2026_01_07_145913) |
| ALE/Robotank-v5 | 20.07 | ppo_atari | [ppo_atari_robotank_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_robotank_2026_01_06_183413) |
| ALE/Seaquest-v5 | 1796 | ppo_atari | [ppo_atari_seaquest_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_seaquest_2026_01_06_183440) |
| ALE/Skiing-v5 | -19340 | ppo_atari | [ppo_atari_skiing_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_skiing_2026_01_06_183424) |
| ALE/Solaris-v5 | 2094 | ppo_atari | [ppo_atari_solaris_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_solaris_2026_01_06_192643) |
| ALE/SpaceInvaders-v5 | 726 | ppo_atari | [ppo_atari_spaceinvaders_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_spaceinvaders_2026_01_07_102346) |
| ALE/StarGunner-v5 | 47495 | ppo_atari_lam70 | [ppo_atari_lam70_stargunner_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_stargunner_2026_01_07_111404) |
| ALE/Surround-v5 | -2.52 | ppo_atari | [ppo_atari_surround_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_surround_2026_01_07_102404) |
| ALE/Tennis-v5 | -4.41 | ppo_atari_lam85 | [ppo_atari_lam85_tennis_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_tennis_2026_01_07_223532) |
| ALE/TimePilot-v5 | 4668 | ppo_atari | [ppo_atari_timepilot_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_timepilot_2026_01_07_101010) |
| ALE/Tutankham-v5 | 217 | ppo_atari_lam85 | [ppo_atari_lam85_tutankham_2026_01_08](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_tutankham_2026_01_08_095251) |
| ALE/UpNDown-v5 | 182472 | ppo_atari | [ppo_atari_upndown_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_upndown_2026_01_07_105708) |
| ALE/VideoPinball-v5 | 56746 | ppo_atari_lam70 | [ppo_atari_lam70_videopinball_2026_01_07](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_videopinball_2026_01_07_224359) |
| ALE/WizardOfWor-v5 | 5814 | ppo_atari | [ppo_atari_wizardofwor_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_wizardofwor_2026_01_06_221154) |
| ALE/YarsRevenge-v5 | 17120 | ppo_atari | [ppo_atari_yarsrevenge_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_yarsrevenge_2026_01_06_221154) |
| ALE/Zaxxon-v5 | 10756 | ppo_atari | [ppo_atari_zaxxon_2026_01_06](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_zaxxon_2026_01_06_221154) |

</details>

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
