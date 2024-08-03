import shutil
import pandas as pd
import os.path as osp
import argparse
import torch
import yaml
from src.types import ModelType, AttackType
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


def run_nti_attack(args, img_idx, prompt, pipe, generator, device):
    image = Image.open(f"{args.out_dir}/{img_idx}/original.png")

    image = image.resize((512, 512))

    pp_image = preprocess_image(image)

    latent = (
        pipe.vae.encode(pp_image.to(device)).latent_dist.sample(generator=generator)
        * pipe.vae.scaling_factor
    )
    # text_embeddings = pipe._encode_prompt(prompt, device, 1, False, None)
    text_embeddings = pipe.encode_prompt(prompt, device, 1, False, None)[0]

    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    all_latents = ddim_inversion(latent, text_embeddings, pipe.scheduler, pipe.unet)

    z_T, all_null_texts = null_text_inversion(
        pipe, all_latents, prompt, num_opt_steps=args.nti_num_opt_steps, device=device
    )
    recon_img = reconstruct(
        pipe, z_T, prompt, all_null_texts, guidance_scale=1, device=device
    )
    recon_img = pipe.image_processor.numpy_to_pil(recon_img)[0]
    recon_img.save(f"{args.out_dir}/{img_idx}/output.png")
    plot_me(pp_image, recon_img, save_path=f"{args.out_dir}/{img_idx}/comparison.png")


def main(args):
    prompts_df = pd.read_csv(osp.join(args.dataset_dir, "prompts.csv"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline(args)
    pipe.unet.load_state_dict(torch.load(args.model_path))
    pipe.to(device)
    for img_idx, prompt in zip(
        prompts_df["case_number"].tolist(), prompts_df["prompt"].tolist()
    ):
        os.makedirs(f"{args.out_dir}/{img_idx}", exist_ok=True)
        shutil.copy(
            osp.join(args.dataset_dir, "imgs", f"{img_idx}_0.png"),
            f"{args.out_dir}/{img_idx}/original.png",
        )
        generator = torch.manual_seed(args.seed)
        if args.attack_type == AttackType.NTI:
            run_nti_attack(args, img_idx, prompt, pipe, generator, device)
        else:
            raise ValueError(f"Attack type {args.attack_type} is not supported")


def validate_and_get_args():
    parser = argparse.ArgumentParser(
        description="Run attack benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # TODO: add choice for model base , from SD1-4, SD2, SD2.1
    parser.add_argument("--model_base", type=str, default="CompVis/stable-diffusion-v1-4", help="Base SD model")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="files/dataset/sd_1_4/nudity",
        help="Path to the dataset base dir. path is relative to --base_dir. datasets can be generated using generate_all_datasets.py",
    )
    parser.add_argument('--num_inference_steps', type=int, default=25, help='Number of inference steps')
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
        choices=ModelType.get_all_types(),
    )
    parser.add_argument(
        "--attack_type",
        type=AttackType.get_associated_type,
        default=AttackType.NTI,
        help="Attack type",
        choices=AttackType.get_all_types(),
    )
    parser.add_argument(
        "--nti_num_opt_steps",
        default=15,
        type=int,
        help="Number of optimization steps for NTI",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    args.dataset_dir = osp.join(args.base_dir, args.dataset_dir)
    assert osp.isdir(
        args.dataset_dir
    ), f"datasets directory {args.dataset_dir} does not exist"

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
    args.out_dir = osp.join(args.base_dir, "files", "results", args.model_type.name, osp.basename(args.dataset_dir))

    os.makedirs(args.out_dir, exist_ok=True)
    save_args = vars(args).copy()
    save_args["model_type"] = args.model_type.name
    save_args["attack_type"] = args.attack_type.name
    args_yaml = osp.join(args.out_dir, "args.yaml")
    with open(args_yaml, "w") as f:
        yaml.dump(save_args, f)
    return args


if __name__ == "__main__":
    args = validate_and_get_args()
    main(args)
