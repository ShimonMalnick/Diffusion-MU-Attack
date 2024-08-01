import pandas as pd
import os.path as osp
import argparse
import torch
import yaml
from src.types import DataType, ModelType, AttackType
import os
from easydict import EasyDict
from src.utils import load_pipeline, preprocess_image
from src.nti import ddim_inversion, null_text_inversion, reconstruct, plot_me
from PIL import Image


def check_if_experiment_exists(args, img_idx):
    # FIXME: implement according to outputs
    dir_root = osp.join(args.out_dir, img_idx)
    conditions = [
        osp.isdir(dir_root),
        osp.isfile(osp.join(dir_root, "input.png")),
        osp.isfile(osp.join(dir_root, "output.png")),
        osp.isfile(osp.join(dir_root, "comparison.png")),
    ]
    if all(conditions):
        return True
    else:
        return False


def run_nti_attack(args, img_idx, generated_image, prompt, pipe, generator, device):
    if generated_image is None:
        generated_image = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
            generator=generator,
        ).images[0]
        generated_image_path = f"{args.out_dir}/{img_idx}/input.png"
        generated_image.save(generated_image_path)

    image = generated_image.resize((512, 512))

    pp_image = preprocess_image(image)

    latent = (
        pipe.vae.encode(pp_image.to(device)).latent_dist.sample(generator=generator)
        * pipe.vae.scaling_factor
    )

    text_embeddings = pipe._encode_prompt(prompt, device, 1, False, None)

    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    all_latents = ddim_inversion(latent, text_embeddings, pipe.scheduler, pipe.unet)

    z_T, all_null_texts = null_text_inversion(
        pipe, all_latents, prompt, num_opt_steps=15, device=device
    )
    recon_img = reconstruct(
        pipe, z_T, prompt, all_null_texts, guidance_scale=1, device=device
    )
    recon_img.save(f"{args.out_dir}/{img_idx}/output.png")
    plot_me(pp_image, recon_img, save_path=f"{args.out_dir}/{img_idx}/comparison.png")


def main(args):
    prompts_df = pd.read_csv(args.prompts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline(args).to(device)
    for img_idx, prompt in zip(
        prompts_df["Unnamed: 0"].tolist(), prompts_df["prompt"].tolist()
    ):
        if check_if_experiment_exists(args, img_idx):
            continue
        if osp.isdir(osp.join(args.out_dir, img_idx)) and osp.isfile(osp.join(args.out_dir, img_idx, "input.png")):
            # skip image generation if already exists
            input_img = Image.open(osp.join(args.out_dir, img_idx, "input.png"))
        else:
            input_img = None
        os.makedirs(f"{args.out_dir}/{img_idx}", exist_ok=True)
        generator = torch.manual_seed(args.seed)
        if args.attack_type == AttackType.NTI:
            run_nti_attack(args, img_idx, input_img, prompt, pipe, generator, device)
        else:
            raise ValueError(f"Attack type {args.attack_type} is not supported")


def validate_and_get_args():
    parser = argparse.ArgumentParser(
        description="Run attack benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="files/dataset/i2p_nude/prompts.csv",
        help="Path to the prompts csv file. path is relative to --base_dir",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/shimon/research/Diffusion-MU-Attack",
        help="Base directory",
    )
    parser.add_argument(
        "--model_type",
        type=ModelType.get_associated_type,
        default=ModelType.EraseDiff,
        help="Model type",
        choices=ModelType.get_all_types_names(),
    )
    parser.add_argument(
        "--attack_type",
        type=AttackType.get_associated_type,
        default=AttackType.NTI,
        help="Attack type",
        choices=AttackType.get_all_types_names(),
    )
    parser.add_argument(
        "--data_type",
        type=DataType.get_associated_type,
        default=DataType.NUDITY,
        help="Data type",
        choices=DataType.get_all_types_names(),
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    args.prompts = osp.join(args.base_dir, args.prompts)
    assert osp.exists(args.prompts), f"prompts file {args.prompts} does not exist"

    args = EasyDict(vars(args))
    default_model_root = "/home/shimon/research/concepts_erasure_models/checkpoints/Baselines/localscratch/damon2024/DM_baselines/ALL_baseline_ckpt/nudity/"
    args.model_root = os.environ.get("MODEL_ROOT", default_model_root)
    # TODO: move model types path to types and configs
    method_to_model_path = {
        "EraseDiff": osp.join(
            args.model_root, "EraseDiff", "EraseDiff-Nudity-Diffusers-UNet.pt"
        ),
        "ESD": osp.join(args.model_root, "ESD", "ESD-Nudity-Diffusers-UNet-noxattn.pt"),
        "FMN": osp.join(args.model_root, "FMN", "FMN-Nudity-Diffusers-UNet.pt"),
        "Salun": osp.join(args.model_root, "Salun", "Salun-Nudity-Diffusers-UNet.pt"),
        "Scissorhands": osp.join(
            args.model_root, "Scissorhands", "Scissorhands-Nudity-Diffusers-UNet.pt"
        ),
        "SPM": osp.join(args.model_root, "SPM", "SPM-Nudity-Diffusers-UNet.pt"),
        "UCE": osp.join(args.model_root, "UCE", "UCE-Nudity-Diffusers-UNet.pt"),
    }
    args.model_path = method_to_model_path[args.model_type.name]
    args.out_dir = osp.join(args.base_dir, "results", args.model_type.name)

    os.makedirs(args.out_dir, exist_ok=True)
    save_args = vars(args).copy()
    save_args["data_type"] = args.data_type.name
    save_args["model_type"] = args.model_type.name
    save_args['attack_type'] = args.attack_type.name
    args_yaml = osp.join(args.out_dir, "args.yaml")
    with open(args_yaml, "w") as f:
        yaml.dump(save_args, f)
    return args


if __name__ == "__main__":
    args = validate_and_get_args()
    main(args)


df = pd.read_csv(
    "/home/shimon/research/Diffusion-MU-Attack/files/dataset/i2p_nude/prompts.csv"
)
images_root = "/home/shimon/research/Diffusion-MU-Attack/files/dataset/i2p_nude/imgs/"

methods = ["EraseDiff", "ESD", "FMN", "Salun", "Scissorhands", "SPM", "UCE"]
for method in methods:
    commands = []
    for img_idx, prompt in zip(df["Unnamed: 0"].tolist(), df["prompt"].tolist()):
        commands.append(
            f'python concept-ablation-diffusers/attack_tamed_model_variable_nudity.py --method {method} --prompt "{prompt}" --image_path {osp.join(images_root, str(img_idx) + "_0.png") } '
        )
        print(commands[-1])

    with open(f"run_nudity_benchmark_{method}.sh", "w") as f:
        f.write("\n".join(commands))


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
for image_idx in df["Unnamed: 0"].tolist():
    path = f"/home/shimon/research/diffusion_concepts/concept-ablation-space/attack_output_ESD_image_{image_idx}_0_2.png"
    all_paths.append(path)
from src.tasks.utils.metrics.nudity_eval import detectNudeClasses, if_nude

all_paths_that_exist = [p for p in all_paths if osp.exists(p)]
for path in all_paths_that_exist:
    res = detectNudeClasses([path])
    print(path.split("_")[-3], if_nude(res[0]))


import os, json
from tqdm import tqdm

methods = ["EraseDiff", "ESD", "FMN", "Salun", "Scissorhands", "SPM", "UCE"]
results = {}
for method in methods:
    root = f"/home/shimon/research/storage/jevnisek/diffusion/results/{method}/"
    image_paths = [
        osp.join(root, x, f"attack_input_{method}_image#1.png")
        for x in os.listdir(root)
    ]
    results[method] = {}
    results[method]["before"] = {}
    all_paths_that_exist = [p for p in image_paths if osp.exists(p)]
    pbar = tqdm(all_paths_that_exist)
    for path in pbar:
        res = detectNudeClasses([path])
        # print(path.split('_')[-3], if_nude(res[0]))
        key = osp.basename(osp.dirname(path))
        results[method]["before"][key] = if_nude(res[0])
        pbar.set_description(f"method {method} | before {key} | {if_nude(res[0])}")

    results[method]["after"] = {}
    image_paths = [
        osp.join(root, x, f"attack_output_{method}_image_{x}_2.png")
        for x in os.listdir(root)
    ]
    all_paths_that_exist = [p for p in image_paths if osp.exists(p)]
    pbar = tqdm(all_paths_that_exist)
    for path in pbar:
        res = detectNudeClasses([path])
        # print(path.split('_')[-3], if_nude(res[0]))
        key = osp.basename(osp.dirname(path))
        results[method]["after"][key] = if_nude(res[0])
        pbar.set_description(f"method {method} | after {key} | {if_nude(res[0])}")
    asr = 0
    for k in results[method]["after"]:
        asr += results[method]["after"][k]
    asr = asr / len(results[method]["after"])
    print(f"[AFTER NTI] method {method} | asr {asr}")
    asr = 0
    for k in results[method]["before"]:
        asr += results[method]["before"][k]
    asr = asr / len(results[method]["before"])
    print(f"[ORIGINAL IMAGE] method {method} | asr {asr}")

    print(f'{set(results[method]["after"])-set(results[method]["before"].keys())}')

    asr = 0
    for k in results[method]["after"]:
        if k in results[method]["before"] and results[method]["before"][k]:
            asr += results[method]["after"][k]
    asr = asr / len(
        [x for x in results[method]["before"] if results[method]["before"][x]]
    )
    print(f"[ASR ON ORIGINAL NUDE IMAGES] method {method} | asr {asr}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
