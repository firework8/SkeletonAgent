# SkeletonAgent

[![paper](https://img.shields.io/badge/arXiv-Paper-red.svg)]() [![models](https://img.shields.io/badge/Link-Models-87CEEB.svg)](https://drive.google.com/drive/folders/1CFMuBrJGktxHfv2GeX8bbgnDuINVrC08?usp=sharing) [![video](https://img.shields.io/badge/License-MIT-yellow?style=flat)](/LICENSE)

This is the official PyTorch implementation for "[SkeletonAgent: An Agentic Interaction Framework for Skeleton-based Action Recognition]()".

### Abstract
> Recent advances in skeleton-based action recognition increasingly leverage semantic priors from Large Language Models (LLMs) to enrich skeletal representations. However, the LLM is typically queried in isolation from the recognition model and receives no performance feedback. As a result, it often fails to deliver the targeted discriminative cues critical to distinguish similar actions. To overcome these limitations, we propose SkeletonAgent, a novel framework that bridges the recognition model and the LLM through two cooperative agents, i.e., Questioner and Selector. Specifically, the Questioner identifies the most frequently confused classes and supplies them to the LLM as context for more targeted guidance. Conversely, the Selector parses the LLM’s response to extract precise joint-level constraints and feeds them back to the recognizer, enabling finer-grained cross-modal alignment. Comprehensive evaluations on five benchmarks, including NTU RGB+D, NTU RGB+D 120, Kinetics-Skeleton, FineGYM, and UAV-Human, demonstrate that SkeletonAgent consistently outperforms state-of-the-art benchmark methods.

## :art: Installation

```shell
git clone https://github.com/firework8/SkeletonAgent.git
cd SkeletonAgent
conda env create -f skeletonagent.yaml
conda activate skeletonagent
pip install -r requirements.txt
pip install -e .
```

## :memo: Data Preparation

PYSKL provides links to the pre-processed skeleton pickle annotations.

- NTU RGB+D: [NTU RGB+D Download Link](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl)
- NTU RGB+D 120: [NTU RGB+D 120 Download Link](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl)
- Kinetics-Skeleton: [Kinetics-Skeleton Download Link](https://download.openmmlab.com/mmaction/pyskl/data/k400/k400_hrnet.pkl)
- FineGYM: [FineGYM Download Link](https://download.openmmlab.com/mmaction/pyskl/data/gym/gym_hrnet.pkl)
- UAV-Human: [UAV-Human Download Link](https://drive.google.com/file/d/1wsejIXMCh9Ip0V_6KdZnajF4LcZos-Y5/view?usp=sharing)


For Kinetics-Skeleton, since the skeleton annotations are large, please use the [Kinetics Annotation Link](https://www.dropbox.com/scl/fi/5phx0m7bok6jkphm724zc/kpfiles.zip?rlkey=sz26ljvlxb6gwqj5m9jvynpg8&st=47vcw2xb&dl=0) to download the `kpfiles` and extract it under `$SkeletonAgent/data/k400` for Kinetics-Skeleton. The `kpfiles` needs to be extracted under `Linux`. Kinetics-Skeleton requires the dependency `Memcached`, which could be referred to [here](https://www.runoob.com/memcached/memcached-install.html). 

You could check the official [Data Doc](https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md) of PYSKL for more detailed instructions.

Notably, [API information](/skeletonagent/models/heads/agent.py#L10) should be configured in the files. Alternatively, you could utilize the [semi-version](/configs/ntu60_xsub/j.py#L15) to train.

## :dizzy: Training & Testing

Please change the config file depending on what you want. You could use the following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.

```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# For example: train on NTU RGB+D X-Sub (Joint Modality) with 1 GPU, with validation, and test the checkpoint.
bash tools/dist_train.sh configs/ntu60_xsub/j.py 1 --validate --test-last --test-best
```

```shell
# Testing
bash tools/dist_test.sh {config_name} {checkpoint_file} {num_gpus} {other_options}
# For example: test on NTU RGB+D X-Sub (Joint Modality) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/ntu60_xsub/j.py checkpoints/CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```

```shell
# Ensemble the results
cd tools
python ensemble.py
```

## :file_folder: Pretrained Models

All the checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/1CFMuBrJGktxHfv2GeX8bbgnDuINVrC08?usp=sharing).

For the detailed performance of pretrained models, please go to the [Model Doc](/data/README.md).

## :beers: Acknowledgements

This repo is mainly based on [PYSKL](https://github.com/kennymckormick/pyskl). We also refer to [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN), [LA-GCN](https://github.com/damnull/lagcn), and [GAP](https://github.com/MartinXM/GAP).

Thanks to the original authors for their excellent work!

## :e-mail: Contact

For any questions, feel free to contact: `hongda.liu@cripac.ia.ac.cn`
