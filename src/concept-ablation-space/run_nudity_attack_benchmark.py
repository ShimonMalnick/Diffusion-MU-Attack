import pandas as pd
import os.path as osp


df = pd.read_csv('/home/shimon/research/Diffusion-MU-Attack/files/dataset/i2p_nude/prompts.csv')
images_root = '/home/shimon/research/Diffusion-MU-Attack/files/dataset/i2p_nude/imgs/'

methods = ['EraseDiff', 'ESD', 'FMN', 'Salun', 'Scissorhands', 'SPM', 'UCE']
for method in methods:
    commands = []
    for img_idx, prompt in zip(df['Unnamed: 0'].tolist(), df['prompt'].tolist()):
        commands.append(f'python concept-ablation-diffusers/attack_tamed_model_variable_nudity.py --method {method} --prompt "{prompt}" --image_path {osp.join(images_root, str(img_idx) + "_0.png") } ')
        print(commands[-1])

    with open(f'run_nudity_benchmark_{method}.sh', 'w') as f:
        f.write('\n'.join(commands))


for method in methods:
    print(f"bash run_nudity_benchmark_{method}.sh")

"""
bash run_nudity_benchmark_EraseDiff.sh
bash run_nudity_benchmark_Scissorhands.sh
bash run_nudity_benchmark_Salun.sh
bash run_nudity_benchmark_ESD.sh
bash run_nudity_benchmark_UCE.sh
bash run_nudity_benchmark_FMN.sh
bash run_nudity_benchmark_SPM.sh


"""


path = "/home/shimon/research/diffusion_concepts/concept-ablation-space/attack_output_ESD_image_296_0_2.png"
all_paths = []
for image_idx in df['Unnamed: 0'].tolist():
    path = f"/home/shimon/research/diffusion_concepts/concept-ablation-space/attack_output_ESD_image_{image_idx}_0_2.png"
    all_paths.append(path)
from src.tasks.utils.metrics.nudity_eval import detectNudeClasses, if_nude

all_paths_that_exist = [p for p in all_paths if osp.exists(p)]
for path in all_paths_that_exist:
    res = detectNudeClasses([path])
    print(path.split('_')[-3], if_nude(res[0]))
