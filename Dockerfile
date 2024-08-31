# set base image
FROM continuumio/miniconda3

# set working directory
WORKDIR /app

# install packages
COPY requirements.txt .
RUN apt-get update \
    && apt install curl -y
# add channels, last added is with the highest priorety
RUN conda config --add channels pytorch
RUN conda config --add channels conda-forge
RUN conda config --add channels anaconda
# install pip for fallback
RUN conda install --yes pip
# install with conda, install with pip on failure
RUN while read requirement; do conda install --yes $requirement \
    || pip install $requirement; done < requirements.txt

# set environment variables
ENV PYTHONPATH /app

# download data
RUN curl -L https://api.github.com/repos/janzing/janzing.github.io/tarball \
                    | tar xz --wildcards "*/code/data_from_optical_device" \
                             --strip-components=2
RUN mkdir --parents data/linear; mv data_from_optical_device/* data/linear

# copy source code
COPY . .

# run script
CMD ["python3", "-u", "src/main.py"]