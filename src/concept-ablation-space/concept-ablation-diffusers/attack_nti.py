import os
from diffusers import StableDiffusionPipeline, DDIMScheduler
import yaml
from model_pipeline import CustomDiffusionPipeline
from transformers import CLIPTextModel
import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import argparse


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
        choices=["object", "nudity"],
        help="Type of attack, whether on an object or nudity",
    )
    parser.add_argument(
        "--object_name",
        type=str,
        required=False,
        choices=["church", "grabage_truck", "parachute", "tench", "van_gogh"],
        help="Object name",
    )
    parser.add_argument("--eta", type=float, default=1.0, help="Eta")
    parser.add_argument(
        "--prompts", type="str", nargs="+", required=True, help="Prompts"
    )
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


def run_on_image(prompt, prompt_idx, pipe, args, device, generator=None):
    generated_image = pipe(
        prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
        generator=generator,
    ).images[0]
    generated_image_path = f"{args.out_dir}/attack_output_{prompt_idx}.png"
    generated_image.save(generated_image_path)
    # TODO fix choices here when adding more types of attacks
    assert (
        args.defenese_method == "advunlearn"
    ), "Currently only advunlearn is supported"
    # load advunlearn result:
    cache_path = "/home/shimon/research/Diffusion-MU-Attack/cache_dir/"
    text_encoder = CLIPTextModel.from_pretrained(
        "OPTML-Group/AdvUnlearn", subfolder="church_unlearned", cache_dir=cache_path
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
        pp_image, recon_img, save_path=f"{args.out_dir}/attack_output_{prompt_idx}.png"
    )


def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image


@torch.no_grad()
def ddim_inversion(latents, encoder_hidden_states, noise_scheduler, unet):
    next_latents = latents
    all_latents = [latents.detach().cpu()]

    # since we are adding noise to the image, we reverse the timesteps list to start at t=0
    reverse_timestep_list = reversed(noise_scheduler.timesteps)

    for i in range(len(reverse_timestep_list) - 1):
        timestep = reverse_timestep_list[i]
        next_timestep = reverse_timestep_list[i + 1]
        latent_model_input = noise_scheduler.scale_model_input(next_latents, timestep)
        noise_pred = unet(latent_model_input, timestep, encoder_hidden_states).sample

        alpha_prod_t = noise_scheduler.alphas_cumprod[timestep]
        alpha_prod_t_next = noise_scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_next = 1 - alpha_prod_t_next

        f = (next_latents - beta_prod_t**0.5 * noise_pred) / (alpha_prod_t**0.5)
        next_latents = alpha_prod_t_next**0.5 * f + beta_prod_t_next**0.5 * noise_pred
        all_latents.append(next_latents.detach().cpu())

    return all_latents


def null_text_inversion(
    pipe,
    all_latents,
    prompt,
    num_opt_steps=15,
    lr=0.01,
    tol=1e-5,
    guidance_scale=7.5,
    eta: float = 0.0,
    generator=None,
    device=None,
):
    # initialise null text embeddings
    null_text_prompt = ""
    null_text_input = pipe.tokenizer(
        null_text_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncaton=True,
        return_tensors="pt",
    )

    # prepare for optimising
    null_text_embeddings = torch.nn.Parameter(
        pipe.text_encoder(null_text_input.input_ids.to(pipe.device))[0],
        requires_grad=True,
    )
    null_text_embeddings = null_text_embeddings.detach()
    null_text_embeddings.requires_grad_(True)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        [null_text_embeddings],  # only optimize the embeddings
        lr=lr,
    )

    # step_ratio = pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps
    text_embeddings = pipe._encode_prompt(prompt, device, 1, False, None).detach()
    # input_embeddings = torch.cat([null_text_embeddings, text_embeddings], dim=0)
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
    all_null_texts = []
    latents = all_latents[-1]
    latents = latents.to(pipe.device)
    for timestep, prev_latents in pipe.progress_bar(
        zip(pipe.scheduler.timesteps, reversed(all_latents[:-1]))
    ):
        prev_latents = prev_latents.to(pipe.device).detach()

        # expand the latents if we are doing classifier free guidance
        latent_model_input = pipe.scheduler.scale_model_input(
            latents, timestep
        ).detach()
        noise_pred_text = pipe.unet(
            latent_model_input, timestep, encoder_hidden_states=text_embeddings
        ).sample.detach()
        for _ in range(num_opt_steps):
            # predict the noise residual
            noise_pred_uncond = pipe.unet(
                latent_model_input, timestep, encoder_hidden_states=null_text_embeddings
            ).sample

            # perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            prev_latents_pred = pipe.scheduler.step(
                noise_pred, timestep, latents, **extra_step_kwargs
            ).prev_sample
            loss = torch.nn.functional.mse_loss(prev_latents_pred, prev_latents).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        all_null_texts.append(null_text_embeddings.detach().cpu())
        latents = prev_latents_pred.detach()
    return all_latents[-1], all_null_texts


@torch.no_grad()
def reconstruct(
    pipe,
    latents,
    prompt,
    null_text_embeddings,
    guidance_scale=7.5,
    generator=None,
    eta=0.0,
    device=None,
):
    text_embeddings = pipe._encode_prompt(prompt, device, 1, False, None)
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
    latents = latents.to(pipe.device)
    for i, (t, null_text_t) in enumerate(
        pipe.progress_bar(zip(pipe.scheduler.timesteps, null_text_embeddings))
    ):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        input_embedding = torch.cat([null_text_t.to(pipe.device), text_embeddings])
        # predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=input_embedding
        ).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs
        ).prev_sample

    # Post-processing
    image = pipe.decode_latents(latents)
    return image


def plot_me(pp_image, recon_img, save_path):
    plt.close("all")
    plt.subplot(1, 3, 1)
    plt.imshow(pp_image[0].permute(1, 2, 0).numpy() * 0.5 + 0.5)
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(recon_img[0])
    plt.title("Reconstructed using Null Text Inversion")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(
        np.abs(recon_img[0] - (pp_image[0].permute(1, 2, 0).numpy() * 0.5 + 0.5)).mean(
            axis=-1
        )
        * 255,
        cmap="grey",
    )
    plt.colorbar()
    plt.title("Gray Scale Diff \nbetween original and Reconstructed")
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches((12, 4))
    plt.tight_layout()
    plt.savefig(save_path)


def main():
    has_cuda = torch.cuda.is_available()
    device = torch.device("cpu" if not has_cuda else "cuda")
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # save args yaml to out_dir
    with open(f"{args.out_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    pipe = load_pipeline(args, device)
    prompts = args.prompts
    generator = torch.manual_seed(args.seed)
    global_out_dir = args.out_dir
    for prompt_idx, prompt in enumerate(prompts):
        args.out_dir = f"{global_out_dir}/prompt_{prompt_idx}"
        os.makedirs(args.out_dir, exist_ok=True)
        run_on_image(prompt, prompt_idx, pipe, args, device, generator)


if __name__ == "__main__":
    main()
