# build:
```shell
cd /home/shimon/research/Diffusion-MU-Attack
docker build -t shimon_run_attacks -f docker/base/Dockerfile .
docker tag shimon_run_attacks shimonmal/run_attacks:base
docker push shimonmal/run_attacks:base
```

# run locally:
```shell
docker run -it --gpus all   -e EXPORT_SCRIPT=name.sh   -e RUN_SCRIPT=say_hello.sh   -v $(pwd)/v5/:/storage/malnick   diffusion_v5_shimon
```

# runai:
```shell
runai submit --name shimon-attack -g 1.0 -i shimonmal/run_attacks:base --pvc=storage:/storage --large-shm -e HF_HOME=/storage/malnick/huggingface_cache -e OUTPUT_ROOT=/storage/malnick/concept_attacks/outputs/ -e MODELS_ROOT=/storage/malnick/concept_attacks/concept_models/ -e DATASETS_ROOT=/storage/malnick/concept_attacks/concept_datasets/ --command -- /bin/bash /storage/malnick/run_script.sh
```

# interactive:
```shell
runai submit --name shimon-attack -g 1.0 -i shimonmal/run_attacks:base --pvc=storage:/storage --large-shm -e HF_HOME=/storage/malnick/huggingface_cache -e OUTPUT_ROOT=/storage/malnick/concept_attacks/outputs/ -e MODELS_ROOT=/storage/malnick/concept_attacks/concept_models/ -e DATASETS_ROOT=/storage/malnick/concept_attacks/concept_datasets/ --command -- sleep infinity
```


