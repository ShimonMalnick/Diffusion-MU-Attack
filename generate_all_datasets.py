import os
import tqdm


if __name__ == "__main__":
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
        os.makedirs(f"files/dataset/sd_1_4/{prompt_name}", exist_ok=True)
        command = f"python src/execs/generate_dataset.py --prompts_path prompts/{prompt_file} --concept {prompt_name} --save_path files/dataset/sd_1_4 --device cuda:0"

        os.system(command)
