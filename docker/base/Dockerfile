FROM shimonmal/diffusion_base_shimon:base


# Install basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    vim \
    nano \
    less \
    git \
    bash-completion \
    screen \
    && rm -rf /var/lib/apt/lists/*

# Create an alias for ll
RUN echo "alias ll='ls -l --color=auto'" >> /root/.bashrc

COPY ./src /code/Diffusion-MU-Attack/src
COPY ./run_attack_benchmark.py /code/Diffusion-MU-Attack/run_attack_benchmark.py
COPY ./run_script.sh /code/Diffusion-MU-Attack/run_script.sh
COPY ./running_pool.py /code/Diffusion-MU-Attack/running_pool.py
COPY ./run_attack_all_datasets.py /code/Diffusion-MU-Attack/run_attack_all_datasets.py
COPY ./configs /code/Diffusion-MU-Attack/configs
COPY ./scripts /code/Diffusion-MU-Attack/scripts
COPY ./prompts /code/Diffusion-MU-Attack/prompts

# CMD ["/bin/bash", "-c", "cd /code/Diffusion-MU-Attack && source run_script.sh"]
