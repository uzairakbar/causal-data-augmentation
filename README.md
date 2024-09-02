# Causal data augmentation

## Setup
### Clone
Clone this repository.
```bash
git clone https://github.com/uzairakbar/causal-data-augmentation.git
```

### Python `venv`
Setup the python virtual environemnt (requires python `3.10.14`).
```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

Then run the main script `.env/bin/python src/main.py`.

### Conda environment
Install dependencies with `conda` and fallback to `pip` if needed.
```bash
environment=causal-data-augmentation
conda create --name "$environment" python=3.10.14 --yes
conda activate "$environment"
conda install pip --yes
pip install -r requirements.txt
```
Then run experiments with `conda run -n causal-data-augmentation python src/main.py`.

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

## Versioning
This project is versioned with #submissions/revisions, #experiments updated (per revision) and #commits to `main`.
```
<#Submissions>.<#UpdatedExperiments>.<#MainCommits>
```
