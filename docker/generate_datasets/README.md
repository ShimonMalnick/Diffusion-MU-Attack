# build:
```shell
cd /home/shimon/research/Diffusion-MU-Attack
docker build -t shimon_generate_datasets -f docker/generate_datasets/Dockerfile .
docker tag shimon_generate_datasets shimonmal/run_attacks:generate_datasets
docker push shimonmal/run_attacks:generate_datasets
```

# run locally (not tested):
```shell
docker run -it --gpus all -e DATASETS_ROOT=/storage/malnick/concept_attacks/concept_datasets/ -v $(pwd)/v5/:/storage/malnick   shimon_generate_datasets
```

# runai:
```shell
runai submit --name shimon-gen-ds -g 1.0 -i shimonmal/run_attacks:generate_datasets --pvc=storage:/storage --large-shm -e HF_HOME=/storage/malnick/huggingface_cache -e DATASETS_ROOT=/storage/malnick/concept_attacks/concept_datasets/ --command -- /bin/bash /storage/malnick/gen_datasets.sh 0|1|2
```

# interactive:
```shell
runai submit --name shimon-gen_ds -g 1.0 -i shimonmal/run_attacks:generate_datasets --pvc=storage:/storage --large-shm -e HF_HOME=/storage/malnick/huggingface_cache -e DATASETS_ROOT=/storage/malnick/concept_attacks/concept_datasets/ --command -- sleep infinity


