# Causal data augmentation

## Setup
### Clone
Clone this repository.
```bash
git clone https://github.com/uzairakbar/causal-data-augmentation.git
```

### Python `venv`
Setup the python virtual environemnt (requires python `3.10.9`).
```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

Then run the main script `.env/bin/python src/main.py`.

### Conda environment
Install dependencies with `conda` and fallback to `pip` if needed.
```bash
conda create --name causal-data-augmentation --yes
conda activate causal-data-augmentation
conda install --yes pip
while read requirement;
do
    conda install -c anaconda \
        -c conda-forge \
        -c pytorch \
        --yes "$requirement" \
    || pip install "$requirement";
done < requirements.txt
```
Then run experiments with `conda run -n causal-data-augmentation python3 ./src/main.py`.

### Docker
Build provided `Dockerfile` and run.
```bash
image=causal-data-augmentation-image
container=causal-data-augmentation-container
docker build -t "$image" .
docker run -v "$PWD"/artifacts:/app/artifacts/ \
    --name "$container" "$image"
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
