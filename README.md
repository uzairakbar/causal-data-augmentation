# Causal Data Augmentation
> Implementation for *"An Analysis of Causal Effect Estimation using Outcome Invariant Data Augmentation"* (NeurIPS 2025).

<p align="center">
    <img src="https://repository-images.githubusercontent.com/555395031/442c67c5-a047-42cf-96a8-1ad7d8587afa"
    alt="Symmetry as Intervention"
    width="42%">
</p>
<p align="center"><i>Symmetry as Intervention; Causal Estimation with Data Augmentation</i></p>
<p align="center">
  <a href="https://arxiv.org/abs/2510.25128"><img src="https://img.shields.io/badge/arXiv-2510.25128-B31B1B.svg?logo" /></a>
  <a href="https://neurips.cc/virtual/2025/poster/119327"><img src="https://img.shields.io/badge/html-%20neurips.cc-8c5cff.svg"></a>
  <a href="https://uzairakbar.github.io/causal-data-augmentation"><img src="https://img.shields.io/badge/WEB-page-0eb077.svg"></a>
  <a href="https://colab.research.google.com/github/uzairakbar/causal-data-augmentation/blob/colab/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

## Contents
- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Citation](#citation)

## Overview

```bash
.
├── src/
│   ├── data_augmentors/    # Data augmentation modules
│   ├── experiments/        # Scripts for running paper experiments
│   ├── regressors/         # ERM, IV, IVL, and baseline models
│   ├── sem/                # Structural equation model definitions
│   └── main.py             # Entry point for training / evaluation
├── config.yaml             # Configuration file for experiments
├── environment.yaml        # Conda environment definition
├── Dockerfile              # Container setup for reproducibility
├── requirements.txt        # Python dependencies
├── LICENSE                 # Code license (MIT)
└── README.md
```

## Setup
Clone this repository.
```bash
git clone https://github.com/uzairakbar/causal-data-augmentation.git
cd causal-data-augmentation
```
To use a GPU, set the `--index-url` value in `requirements.txt` as per your CUDA version (see [PyTorch installation instructions](https://pytorch.org/)).

Proceed setup using one of the below options. Or instead simply [open this project in Colab](https://colab.research.google.com/github/uzairakbar/causal-data-augmentation/blob/colab/demo.ipynb).

### Conda environment (recommended)
```bash
environment=causal-data-augmentation
conda env create -f environment.yaml
conda activate "$environment"
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Python `venv`
Set up the Python virtual environment (requires python `3.10.14`).
```bash
environment='.causal-data-augmentation'
python -m venv "$environment"
"$environment"/bin/python -m pip install -r requirements.txt
source "$environment"/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Docker
Build provided `Dockerfile` and run.
```bash
image=causal-data-augmentation-image
container=causal-data-augmentation-container
docker build --tag "$image" .
docker run --name "$container" \
    --volume "$PWD"/data:/app/data/ \
    --volume "$PWD"/artifacts:/app/artifacts/ \
    "$image"
```

To delete Docker artifacts after finishing experiments, run the following commands.
```bash
image=causal-data-augmentation-image
container=causal-data-augmentation-container
docker rm "$container"
docker image rm -f "$image"
```

## Usage
### Configuring experiments
Use the `./config.yaml` file to specify the experiment parameters. The provided (default) configuration was used to generate the figures of the paper.

Comment out (or remove) the experiments from `./config.yaml` that you are not interested in, and then run the `./src/main.py` script to run the remaining experiments.

The generated figures and artifacts are saved in the `./artifacts/` directory after the experiments finish execution.

### CPU vs. GPU backend
The code uses a CPU backend for PyTorch by default (recommended for `optical_device` and `linear_simulation` experiments). To use a GPU or MPS backend, however, change the `CPU_ONLY` variable specified in `./src/regressors/utils.py` to `False`.

### Running experiments
Simply run the main script `python src/main.py`, or run the Docker container (see above).

## Citation
If you find our work helpful, please consider citing our paper and leaving a star :star:.
```bibtex
@misc{akbar2025causalDataAugmentation,
      title={An Analysis of Causal Effect Estimation using Outcome Invariant Data Augmentation}, 
      author={Uzair Akbar and Niki Kilbertus and Hao Shen and Krikamol Muandet and Bo Dai},
      year={2025},
      eprint={2510.25128},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.25128}, 
}
```

