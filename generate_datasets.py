import os
import pandas as pd
import tqdm
import sys
import os.path as osp


def verify_dataset_creation(datasets_root, model_base_name, prompt_name):
    dir_name = osp.join(datasets_root, model_base_name, prompt_name)
    if not osp.exists(dir_name):
        return False
    prompts_df = osp.join(dir_name, "prompts.csv")
    if not osp.exists(prompts_df):
        return False
    prompts_df = pd.read_csv(prompts_df)
    if len(prompts_df) < len(os.listdir(osp.join(dir_name, "imgs"))):
        return False
    return True
    

def main():
    model_bases = [
        ("CompVis/stable-diffusion-v1-4", "sd_1_4"),
        ("stabilityai/stable-diffusion-2-base", "sd_2"),
        ("stabilityai/stable-diffusion-2-1-base", "sd_2_1"),
    ]
    assert "DATASETS_ROOT" in os.environ, "Please set the DATASETS_ROOT environment"
    datasets_root = os.environ["DATASETS_ROOT"]
    check_already_created = True
    
    
    # change model base according to the model you want to use
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model_base, model_base_name = model_bases[idx]
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    prompt_files = [
        "parachute.csv",
        "tench.csv",
        "vangogh.csv",
        "church.csv",
        "garbage_truck.csv",
        "nudity.csv",
        "violence.csv",
        "illegal.csv",
        "i2p.csv",
        "coco_10k.csv"
    ]

    for prompt_file in tqdm.tqdm(prompt_files):
        prompt_name = prompt_file.replace(".csv", "")
        if check_already_created and verify_dataset_creation(datasets_root, model_base_name, prompt_name):
            print("skipping", model_base_name, prompt_name)
            continue
            
        os.makedirs(f"{osp.join(datasets_root, model_base_name, prompt_name)}", exist_ok=True)
        command = f"python src/execs/generate_dataset.py --model_base {model_base} --prompts_path {osp.join(script_dir, 'prompts', prompt_file)} --concept {prompt_name} --save_path {osp.join(datasets_root, model_base_name)} --device cuda:0"
        os.system(command)
    

if __name__ == "__main__":
    main()
