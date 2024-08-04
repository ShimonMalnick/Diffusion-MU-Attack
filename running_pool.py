import os
import os.path as osp
from src.types import ModelType


data_types_to_dir_names = {
    "Nudity": ["nudity"],
    "Object": ["tench", "parachute", "garbage_truck", "church"],
    "VanGogh": ["vangogh"],
}

    
datasets_root = os.environ['DATASETS_ROOT']
models_root = os.environ['MODELS_ROOT']
output_root = os.environ['OUTPUT_ROOT']
model_bases = [
        ("CompVis/stable-diffusion-v1-4", "sd_1_4"),
        ("stabilityai/stable-diffusion-2-base", "sd_2"),
        ("stabilityai/stable-diffusion-2-1-base", "sd_2_1"),
    ]

def get_next_job():
    for model_base, model_base_name in model_bases:
        for data_type in data_types_to_dir_names:
            for dir_name in data_types_to_dir_names[data_type]:
                for model_type in ModelType.get_all_types_names():
                    if not os.path.exists(f"{output_root}/{model_base_name}/{model_type}/{dir_name}"):
                        return model_base, model_base_name, model_type, dir_name, data_type
    return None

def run_job(model_base, model_base_name, model_type, dir_name, data_type):
    outdir = osp.join(output_root, model_base_name, model_type, dir_name)
    command = f"python run_attack_benchmark.py --model_base {model_base} --dataset_dir {osp.join(datasets_root, model_base_name, dir_name)} --model_type {model_type} --data_type {data_type} --out_dir {outdir}"
    print(f"Runnning command:\n{command}")
    if os.system(command) != 0:
        return False


while True:
    next_job = get_next_job()
    if next_job is None:
        break
    out = run_job(*next_job)
    if not out:
        break
    
