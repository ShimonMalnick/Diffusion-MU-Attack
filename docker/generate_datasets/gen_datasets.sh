#!/bin/bash

echo "Running pool $1"
cd /code/Diffusion-MU-Attack && conda init bash
echo "source activate concept" > ~/.bashrc && source ~/.bashrc
nohup python generate_datasets.py $1 > /storage/malnick/concept_attacks/outputs/logs/gen_datasets_$1.log 2> /storage/malnick/concept_attacks/outputs/logs/gen_datasets_$1.err
