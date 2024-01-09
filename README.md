# Consistency Policy Q-Learning (CPQL)

This is the official PyTorch implementation of the paper "[**Boosting Continuous Control with Consistency Policy**](https://arxiv.org/pdf/2310.06343.pdf)".

# 🛠️ Installation Instructions
## Clone this repository.
```bash
git clone https://github.com/cccedric/cpql.git
cd cpql
```
## Create a virtual environment.
```bash
conda env create -f cpql_env.yaml
```

## Install extra dependencies.
- Install mujoco210 and mujoco-py following instructions [here](https://github.com/openai/mujoco-py#install-mujoco).
- Install D4RL following instructions [here](https://github.com/Farama-Foundation/D4RL).

# 💻 Reproducing Experimental Results
## Training for offline tasks
```bash
python main.py --rl_type offline --env_name hopper-medium-expert-v2
```

## Training for online tasks
```bash
python main.py --rl_type online --env_name Hopper-v3
```

# ✉️ Contact
For any questions, please feel free to email chenyuhui2022@ia.ac.cn.

# 🙏 Acknowledgement
Our code is built upon [comsistency models](https://github.com/openai/consistency_models), [Diffusion-QL](https://github.com/twitter/diffusion-rl). We thank all these authors for their nicely open sourced code and their great contributions to the community.

# 🏷️ License
This repository is released under the GNU license. See [LICENSE](LICENSE) for additional details.

# 📝 Citation
If you find our work useful, please consider citing:
```
@article{chen2023boosting,
  title={Boosting Continuous Control with Consistency Policy},
  author={Chen, Yuhui and Li, Haoran and Zhao, Dongbin},
  journal={arXiv preprint arXiv:2310.06343},
  year={2023}
}
```

























