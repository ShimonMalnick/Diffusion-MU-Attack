import os
import tqdm
import sys


if __name__ == "__main__":
    model_bases = [
        ("CompVis/stable-diffusion-v1-4", "sd_1_4"),
        ("stabilityai/stable-diffusion-2-base", "sd_2"),
        ("stabilityai/stable-diffusion-2-1-base", "sd_2_1"),
    ]
    
    # change model base according to the model you want to use
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model_base, model_base_name = model_bases[idx]
    
    prompt_files = [
        "parachute.csv",
        "tench.csv",
        "vangogh.csv",
        "church.csv",
        "garbage_truck.csv",
        "nudity.csv",
    ]

    for prompt_file in tqdm.tqdm(prompt_files):
        prompt_name = prompt_file.replace(".csv", "")
        os.makedirs(f"files/dataset/{model_base_name}/{prompt_name}", exist_ok=True)
        command = f"python src/execs/generate_dataset.py --prompts_path prompts/{prompt_file} --concept {prompt_name} --save_path files/dataset/{model_base_name} --device cuda:0"

        os.system(command)
