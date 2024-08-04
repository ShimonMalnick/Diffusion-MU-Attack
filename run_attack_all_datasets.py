import os
from src.types import ModelType
import tqdm


if __name__ == "__main__":
    base_dir = "/home/shimon/research/Diffusion-MU-Attack/files/dataset/sd_1_4"
    all_datasets_names = os.listdir(base_dir)
    for ds_name in tqdm.tqdm(all_datasets_names, desc="Datasets"):
        for model_type in tqdm.tqdm(ModelType.get_all_types_names(), desc="Models"):
            command = f"python run_attack_benchmark.py --dataset_dir {base_dir}/{ds_name} --model_type {model_type}"
            print(command)
            os.system(command)