# SSD基准代码
## 概述
这是一份SSD训练的基准代码，支持经典的SSD环境（CleanUp和Harvest）和MeltingPot 2.0环境。
代码主要包括对环境、PPO agent、independent learning的训练流程的实现。
代码支持CNN和RNN，支持多进程采样，包含若干常用trick。

## 组件介绍
### `\ssd_pettingzoo`
包含经典的CleanUp和Harvest环境以及多线程支持组件
代码来自<https://github.com/eugenevinitsky/sequential_social_dilemma_games>。修复了一些bug。

### `\ssd_pettingzoo`
包含meltingpot支持工具以及多线程支持组件。环境本身需要安装[meltingpot 2.0库](https://github.com/google-deepmind/meltingpot)。
支持工具来自<https://github.com/rstrivedi/Melting-Pot-Contest-2023>。

### `main_ppo_clean_up.py`
使用independent PPO训练经典CleanUp环境。

### `main_ppo_clean_up_meltingpot.py`
使用independent PPO训练meltingpot中的CleanUp环境。

### `ppo_agent.py`
PPO实现，支持CNN和RNN。

### `normalization.py`
标准化支持。

### `vectorize_replaybuffer.py`
向量化的经验回放池。

### `utils.py`
一些工具