import os
from diffusers import StableDiffusionPipeline, DDIMScheduler
import yaml
from model_pipeline import CustomDiffusionPipeline
from transformers import CLIPTextModel
import torch
import numpy as np
from matplotlib import pyplot as plt
import argparse


class AttackObject:
    objects_prompts_dict = {"church": 'Church',
                "grabage_truck": 'Garbage_Truck',
                "parachute": 'Parachute',
                "tench": 'Tench',
                "van_gogh": 'VanGogh'}
    
    def __init__(self):
        raise ValueError('This class is not meant to be instantiated')
    
    @staticmethod
    def get_prompt(name):
        AttackObject.validate_name(name)
        return AttackObject.objects_prompts_dict[name]
    
    @staticmethod
    def validate_name(name):
        assert name in AttackObject.objects_prompts_dict, f"Object name {name} not found"
    
        

def get_args():
    parser = argparse.ArgumentParser(
        description="Run NTI attack on model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_base",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Model base",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=6.0, help="Guidance scale"
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    # FIXME add other defense methods
    parser.add_argument(
        "--defenese_method",
        type=str,
        default="advunlearn",
        choices=["advunlearn"],
        help="Defense method",
    )
    parser.add_argument(
        "type",
        type=str,
        #FIXME: add nudity support
        choices=["object"],
        help="Type of attack, whether on an object or nudity",
    )
    parser.add_argument(
        "--object_name",
        type=str,
        choices=AttackObject.objects_prompts_dict.keys(),
        help="Object name",
    )
    parser.add_argument("--eta", type=float, default=1.0, help="Eta")
    parser.add_argument("-seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    # Check if object name is supplied in case the type is object
    assert not (
        args.type == "object" and args.object_name is None
    ), "Object name is required for object type"
    return args


def load_pipeline(args, device=None) -> StableDiffusionPipeline:
    scheduler = DDIMScheduler.from_config(args.model_base, subfolder="scheduler")
    pipe = CustomDiffusionPipeline.from_pretrained(
        args.model_base,
        safety_checker=None,
        scheduler=scheduler,
    )
    if device is not None:
        pipe.to(device)
    return pipe


def run_on_image(prompt, pipe, args, device, generator=None):
    generated_image = pipe(
        prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
        generator=generator,
    ).images[0]
    generated_image_path = f"{args.out_dir}/attack_output.png"
    generated_image.save(generated_image_path)
    # TODO fix choices here when adding more types of attacks
    assert (
        args.defenese_method == "advunlearn"
    ), "Currently only advunlearn is supported"
    # load advunlearn result:
    cache_path = "/home/shimon/research/Diffusion-MU-Attack/cache_dir/"
    text_encoder = CLIPTextModel.from_pretrained(
        "OPTML-Group/AdvUnlearn", subfolder=f"{args.object_type}_unlearned", cache_dir=cache_path
    )
    pipe.text_encoder = text_encoder

    pipe.to(device)

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
    plot_me(
        pp_image, recon_img, save_path=f"{args.out_dir}/attack_output.png"
    )


def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image



def main():
    has_cuda = torch.cuda.is_available()
    device = torch.device("cpu" if not has_cuda else "cuda")
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # save args yaml to out_dir
    with open(f"{args.out_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    pipe = load_pipeline(args, device)
    generator = torch.manual_seed(args.seed)
    # FIXME: currently no nudity support
    prompt = AttackObject.get_prompt(args.object_type)
    run_on_image(prompt, pipe, args, device, generator)


if __name__ == "__main__":
    main()
