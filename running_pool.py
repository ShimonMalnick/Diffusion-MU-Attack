import os
import os.path as osp
from src.types import ModelType
import sys

data_types_to_dir_names = {
    "Nudity": ["nudity"],
    "Object": ["tench", "parachute", "garbage_truck", "church"],
    "VanGogh": ["vangogh"],
}


datasets_root = os.environ["DATASETS_ROOT"]
models_root = os.environ["MODELS_ROOT"]
output_root = os.environ["OUTPUT_ROOT"]
model_bases = [
    ("CompVis/stable-diffusion-v1-4", "sd_1_4"),
    # ("stabilityai/stable-diffusion-2-base", "sd_2"),
    # ("stabilityai/stable-diffusion-2-1-base", "sd_2_1"),
]


def experiment_is_valid(dir_name, model_type, data_type):
    data_name = "_".join([t.title() for t in osp.basename(dir_name).split("_")])
    model_dir_name = osp.join(models_root, data_type.lower(), model_type)
    return osp.exists(model_dir_name) and any(
        [p for p in os.listdir(model_dir_name) if data_name in p]
    )


def get_next_job_params():
    for model_base, model_base_name in model_bases:
        for data_type in data_types_to_dir_names:
            for dir_name in data_types_to_dir_names[data_type]:
                for model_type in ModelType.get_all_types_names():
                    if experiment_is_valid(
                        dir_name, model_type, data_type
                    ) and not osp.exists(
                        f"{output_root}/{model_base_name}/{model_type}/{dir_name}"
                    ):
                        yield (
                            model_base,
                            model_base_name,
                            model_type,
                            dir_name,
                            data_type,
                        )


def run_job(model_base, model_base_name, model_type, dir_name, data_type, dry_run=False):
    outdir = osp.join(output_root, model_base_name, model_type, dir_name)
    command = f"python run_attack_benchmark.py --model_base {model_base} --dataset_dir {osp.join(datasets_root, model_base_name, dir_name)} --model_type {model_type} --data_type {data_type} --out_dir {outdir}"
    print(f"Runnning command:\n{command}")
    if dry_run:
        return True
    if os.system(command) != 0:
        return False

if __name__ == "__main__":
    # all_params = list(get_next_job_params())
    # if len(sys.argv) < 2:
    #     for params in all_params:
    #         run_job(*params, dry_run=False)
    # else:
    #     assert 0 <= int(sys.argv[1]) < len(all_params), f"Invalid index {sys.argv[1]}"
    #     run_job(*all_params[int(sys.argv[1])], dry_run=False)

    while True:
        next_job_params_iterator = get_next_job_params()
        try:
            next_job_params = next(next_job_params_iterator)
        except StopIteration:
            break
        out = run_job(*next_job_params)
        if not out:
            break
