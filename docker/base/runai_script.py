import os
import argparse
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--start', type=int, default=0, help='Start number')
    
    args = parser.parse_args()
    
    # Rest of your code goes here
    for n in range(args.start, args.start + args.runs):
        print("Submitting to runai number: ", n)
        command = f"runai submit --name shimon-attack-a-{n} -g 1.0 -i shimonmal/run_attacks:base --pvc=storage:/storage --large-shm -e HF_HOME=/storage/malnick/huggingface_cache -e OUTPUT_ROOT=/storage/malnick/concept_attacks/outputs/ -e MODELS_ROOT=/storage/malnick/concept_attacks/concept_models/ -e DATASETS_ROOT=/storage/malnick/concept_attacks/concept_datasets/ --command -- /bin/bash /storage/malnick/run_script.sh {n}"
        os.system(command)
        # wait for 3 seconds
        time.sleep(3)