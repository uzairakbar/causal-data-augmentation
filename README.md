# Causal data augmentation

## Setup
### Dependencies
This code was tested on **MacOS** (Apple silicon) and **Linux**. We recommend running this code using `conda`, however we have also provided Docker and python `venv` setup scripts as alternatives.

### Python `venv`
Setup the python virtual environemnt (requires python `3.10.14`).
```bash
environment=causal-data-augmentation
python -m venv "$environment"
source "$environment"/bin/activate
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
Then run the main script `causal-data-augmentation/bin/python src/main.py`.

### Conda environment
Install dependencies with `conda` and fallback to `pip` if needed.
```bash
environment=causal-data-augmentation
conda create --name "$environment" python=3.10.14 --yes
conda activate "$environment"
conda install pip --yes
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
Then run experiments with `conda run --no-capture-output -n causal-data-augmentation python src/main.py`.

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

To delete docker artifacts after finishing experiments, run the following commands.
```bash
image=causal-data-augmentation-image
container=causal-data-augmentation-container
docker rm "$container"
docker image rm -f "$image"
```

## Experiment configuration
Use the `./config.yaml` file to specify the experiment parameters. The provided (default) configuration was used to generate the figures of the paper.

Comment out (or remove) the experiemnts from `./config.yaml` that you are not interested in, and then run the `./src/main.py` script to run the remaining experiments.

The generated figures and artifacts are saved in the `./artifacts/` directory after the experiments finish execution.

## CPU vs. GPU backend
We recommend running the `linear_simulation` and `optical_device` experiments on the CPU. To use a GPU or MPS backend, however, change the `CPU_ONLY` variable specified in `./src/regressors/utils.py` to `False`.
