# syntax=docker/dockerfile:experimental
# Use Python 3.10 as the base image
FROM python:3.10-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y curl gnupg

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk git

WORKDIR /app
COPY . /app
# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1
RUN pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

RUN pip install 'litgpt[all]' optax fire

RUN pip install --pre torch==2.9.0.dev20250909+cpu torchvision==0.24.0.dev20250909+cpu torchaudio==2.8.0.dev20250909+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
WORKDIR /
RUN git clone https://github.com/pytorch/xla.git
WORKDIR xla/torchax
RUN pip install -e .

RUN pip install flax

ENTRYPOINT ["python", "examples.train_llama.train_llama_lightning"]
CMD ["--batch_size=8", "--seqlen=2048"]