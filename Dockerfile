# set base image
FROM continuumio/miniconda3

# set working directory
WORKDIR /app

# install packages
COPY requirements.txt .
RUN apt-get update \
    && apt install curl -y
RUN conda install -c conda-forge --yes --file requirements.txt

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