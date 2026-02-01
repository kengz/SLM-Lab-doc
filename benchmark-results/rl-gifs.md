---
description: Trained agents in action—watch SLM Lab's PPO and SAC algorithms play games and control robots.
---

# Agent Replays 🎬

These GIFs show trained agents running in "enjoy" mode. Generate your own with:

```bash
slm-lab run _ _ enjoy@data/your_experiment/spec.json
```

## Atari (PPO)

|  |  |
| :---: | :---: |
| ![BeamRider](https://user-images.githubusercontent.com/8209263/63994698-689ecf00-caaa-11e9-991f-0a5e9c2f5804.gif) | ![Breakout](https://user-images.githubusercontent.com/8209263/63994695-650b4800-caaa-11e9-9982-2462738caa45.gif) |
| BeamRider | Breakout |
| ![KungFuMaster](https://user-images.githubusercontent.com/8209263/63994690-60469400-caaa-11e9-9093-b1cd38cee5ae.gif) | ![MsPacman](https://user-images.githubusercontent.com/8209263/63994685-5cb30d00-caaa-11e9-8f35-78e29a7d60f5.gif) |
| KungFuMaster | MsPacman |
| ![Pong](https://user-images.githubusercontent.com/8209263/63994680-59b81c80-caaa-11e9-9253-ed98370351cd.gif) | ![Qbert](https://user-images.githubusercontent.com/8209263/63994672-54f36880-caaa-11e9-9757-7780725b53af.gif) |
| Pong | Qbert |
| ![Seaquest](https://user-images.githubusercontent.com/8209263/63994665-4dcc5a80-caaa-11e9-80bf-c21db818115b.gif) | ![SpaceInvaders](https://user-images.githubusercontent.com/8209263/63994624-15c51780-caaa-11e9-9c9a-854d3ce9066d.gif) |
| Seaquest | SpaceInvaders |

## MuJoCo (SAC)

|  |  |
| :---: | :---: |
| ![Ant](https://user-images.githubusercontent.com/8209263/63994867-ff6b8b80-caaa-11e9-971e-2fac1cddcbac.gif) | ![HalfCheetah](https://user-images.githubusercontent.com/8209263/63994869-01354f00-caab-11e9-8e11-3893d2c2419d.gif) |
| Ant | HalfCheetah |
| ![Hopper](https://user-images.githubusercontent.com/8209263/63994871-0397a900-caab-11e9-9566-4ca23c54b2d4.gif) | ![Humanoid](https://user-images.githubusercontent.com/8209263/63994883-0befe400-caab-11e9-9bcc-c30c885aad73.gif) |
| Hopper | Humanoid |
| ![InvertedDoublePendulum](https://user-images.githubusercontent.com/8209263/63994879-07c3c680-caab-11e9-974c-06cdd25bfd68.gif) | ![InvertedPendulum](https://user-images.githubusercontent.com/8209263/63994880-085c5d00-caab-11e9-850d-049401540e3b.gif) |
| InvertedDoublePendulum | InvertedPendulum |
| ![Reacher](https://user-images.githubusercontent.com/8209263/63994881-098d8a00-caab-11e9-8e19-a3b32d601b10.gif) | ![Walker2d](https://user-images.githubusercontent.com/8209263/63994882-0abeb700-caab-11e9-9e19-b59dc5c43393.gif) |
| Reacher | Walker2d |

{% hint style="info" %}
These replays were generated with SLM Lab v4 using Roboschool. In v5, MuJoCo environments (`Hopper-v5`, `HalfCheetah-v5`, etc.) provide similar physics with improved stability.
{% endhint %}
