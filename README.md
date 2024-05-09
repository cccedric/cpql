# Consistency Policy Q-Learning (CPQL)

This is the official PyTorch implementation of the paper "Boosting Continuous Control with Consistency Policy". For those interested in delving deeper into our research, you can find detailed versions of our paper:

For an extended read, including the appendix, check out the [**Arxiv Version with Appendix**](https://arxiv.org/pdf/2310.06343.pdf).
For the conference-specific details as presented at AAMAS 2024, access the [**AAMAS 2024 Version**]([https://arxiv.org/pdf/2310.06343.pdf](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p335.pdf)).

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
Our code is built upon [consistency models](https://github.com/openai/consistency_models), [Diffusion-QL](https://github.com/twitter/diffusion-rl). We thank all these authors for their nicely open sourced code and their great contributions to the community.

# 🏷️ License
This repository is released under the GNU license. See [LICENSE](LICENSE) for additional details.

# 📝 Citation
If you find our research helpful and would like to reference it in your work, please consider using one of the following citations, depending on the format that best suits your needs:

For the Arxiv version:
```
@article{chen2023boosting,
  title={Boosting Continuous Control with Consistency Policy},
  author={Chen, Yuhui and Li, Haoran and Zhao, Dongbin},
  journal={arXiv preprint arXiv:2310.06343},
  year={2023}
}
```
Or, for citing our work presented at the conference of AAMAS 2024:
```
@inproceedings{DBLP:conf/atal/ChenLZ24,
  author       = {Yuhui Chen and
                  Haoran Li and
                  Dongbin Zhao},
  editor       = {Mehdi Dastani and
                  Jaime Sim{\~{a}}o Sichman and
                  Natasha Alechina and
                  Virginia Dignum},
  title        = {Boosting Continuous Control with Consistency Policy},
  booktitle    = {Proceedings of the 23rd International Conference on Autonomous Agents
                  and Multiagent Systems, {AAMAS} 2024, Auckland, New Zealand, May 6-10,
                  2024},
  pages        = {335--344},
  publisher    = {{ACM}},
  year         = {2024},
  url          = {https://dl.acm.org/doi/10.5555/3635637.3662882},
  doi          = {10.5555/3635637.3662882},
  timestamp    = {Fri, 03 May 2024 14:31:38 +0200},
  biburl       = {https://dblp.org/rec/conf/atal/ChenLZ24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
























