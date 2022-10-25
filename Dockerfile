# set base image
FROM python:3.10.8

# set working directory
WORKDIR /app

# install packages
COPY requirements.txt .
RUN apt-get update
RUN python3 -m pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# set environment variables
ENV PYTHONPATH /app

# copy source code
COPY . .

# run script
CMD ["python3", "src/main.py"]