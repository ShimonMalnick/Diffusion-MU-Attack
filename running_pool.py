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
    # currently there are only models for sd_1_4
    # ("stabilityai/stable-diffusion-2-base", "sd_2"),
    # ("stabilityai/stable-diffusion-2-1-base", "sd_2_1"),
]


def is_valid_experiment(model_base_name, dir_name, model_type, data_type):
    """
    Valid experiments are experiments that have a model associated with them
    """
    model_dir_name = osp.join(models_root, data_type.lower(), model_type)
    if not osp.exists(model_dir_name):
        return False
    data_name = "_".join([t.title() for t in osp.basename(dir_name).split("_")])
    if data_name.lower() == 'vangogh': # for vangogh we need VanGogh although dir name is vangogh and not van_gogh
        data_name = 'VanGogh'
    
    # return True if there is some model in the directory that has this particular name in it
    return any([p for p in os.listdir(model_dir_name) if data_name in p])
    


def experiment_complete(model_base_name, model_type, dir_name, verbose=False):
    """
    Returns False if this experiment has not yet been run successfully, i.e. the output directory does not exist or
    does not contain all expected images. True otherwise.
    """
    imgs_dataset = os.listdir(osp.join(datasets_root, model_base_name, dir_name, "imgs"))
    
    # each image experiment is a directory containing 3 images: original, attack, and comparison
    cur_exp_dir = osp.join(output_root, model_base_name, model_type, dir_name)
    if not osp.exists(cur_exp_dir):
        if verbose:
            print(f"Directory {cur_exp_dir} does not exist")
        return False
    imgs_dirs = [p for p in os.listdir(cur_exp_dir) if osp.isdir(osp.join(cur_exp_dir, p))]
    if len(imgs_dataset) != len(imgs_dirs):
        if verbose:
            print(f"{len(imgs_dirs)}/{len(imgs_dataset)} : not complete. params: {model_base_name},{model_type},{dir_name}")
        return False
    for img_dir in imgs_dirs:
        if len(os.listdir(osp.join(cur_exp_dir, img_dir))) != 3:  # each img dir should have 3 images
            if verbose:
                print(f"Directory {img_dir} in {model_base_name}/{dir_name} has less than 3 images")
            return False
    return True
    


def get_next_job_params(verbose=False):
    for model_base, model_base_name in model_bases:
        for data_type in data_types_to_dir_names:
            for dir_name in data_types_to_dir_names[data_type]:
                for model_type in ModelType.get_all_types_names():
                    if not is_valid_experiment(
                        model_base_name, dir_name, model_type, data_type):
                        continue
                    if not experiment_complete(model_base_name, model_type, dir_name, verbose=verbose):
                        yield (
                            model_base,
                            model_base_name,
                            model_type,
                            dir_name,
                            data_type,
                        )


def get_job_command(model_base, model_base_name, model_type, dir_name, data_type):
    outdir = osp.join(output_root, model_base_name, model_type, dir_name)
    command = f"python run_attack_benchmark.py --model_base {model_base} --dataset_dir {osp.join(datasets_root, model_base_name, dir_name)} --model_type {model_type} --data_type {data_type} --out_dir {outdir}"
    return command


if __name__ == "__main__":
    dry_run = False
    all_params = list(get_next_job_params(verbose=dry_run))
    all_commands = [get_job_command(*params) for params in all_params]
    print(len(all_commands))
    if len(sys.argv) > 1:
        indices = [int(x) for x in sys.argv[1:]]
        assert all([0 <= x < len(all_params) for x in indices]), f"Invalid index {sys.argv[1]}"
        for idx in indices:
            os.system(all_commands[int(sys.argv[1])])
    else:
        print(all_commands)