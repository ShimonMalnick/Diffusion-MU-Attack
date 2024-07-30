import numpy as np
from diffusers import DDIMScheduler
from matplotlib import pyplot as plt

from model_pipeline import CustomDiffusionPipeline
import torch


def preprocess_image(image, device=None):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float16) / 127.5 - 1.0
    if device:
        image = image.to(device)
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

        f = (next_latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5)
        next_latents = alpha_prod_t_next ** 0.5 * f + beta_prod_t_next ** 0.5 * noise_pred
        all_latents.append(next_latents.detach().cpu())

    return all_latents

@torch.no_grad()
def reconstruct(pipe, latents, prompt, null_text_embeddings, guidance_scale=7.5, generator=None, eta=0.0, device=None):
    text_embeddings = pipe._encode_prompt(prompt, device, 1, False, None)
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
    latents = latents.to(pipe.device)
    for i, (t, null_text_t) in enumerate(pipe.progress_bar(zip(pipe.scheduler.timesteps, null_text_embeddings))):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        input_embedding = torch.cat([null_text_t.to(pipe.device), text_embeddings])
        # predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
        # debug: generate the image
        image = pipe.decode_latents(latents)
        import ipdb; ipdb.set_trace()
        plt.close('all')
        plt.imshow(image[0].permute(1, 2, 0).to(dtype=torch.float32).cpu().numpy())
        plt.savefig(f'debug/image_at_iter_{i:03d}.png')

    #Post-processing
    image = pipe.decode_latents(latents)
    return image


@torch.no_grad()
def reconstruct_image_from_latents(all_latents, device):
    model_base = "CompVis/stable-diffusion-v1-4"

    pipe = CustomDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16, safety_checker=None).to(
        device)
    pipe.scheduler = DDIMScheduler.from_config(model_base, subfolder="scheduler")
    result = pipe(prompt=[""] * len(all_latents), latents=torch.cat(all_latents))
    return result

def optimize(image, pipe, device, prompt):
    pp_image = preprocess_image(image, device)

    generator = torch.Generator(device=device)
    latent = pipe.vae.encode(pp_image.to(device)).latent_dist.sample(generator=generator) * 0.18215

    text_embeddings = pipe._encode_prompt(prompt, device, 1, False, None)
    num_inference_steps = 50
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    all_latents = ddim_inversion(latent, text_embeddings, pipe.scheduler, pipe.unet)
    # DDIM inversion debug:
    reconstructed_images = reconstruct_image_from_latents([all_latents[0]], device)
    reconstructed_images[0][0].save('timestamp_0_temp.png')
    reconstructed_images = reconstruct_image_from_latents([all_latents[-1]], device)
    reconstructed_images[0][0].save('timestamp_-1_temp.png')
    reconstructed_images = reconstruct_image_from_latents([all_latents[-10]], device)
    reconstructed_images[0][0].save('timestamp_-10_temp.png')
    reconstructed_images = reconstruct_image_from_latents([all_latents[10]], device)
    reconstructed_images[0][0].save('timestamp_10_temp.png')


    z_T, all_null_texts = null_text_inversion(pipe, all_latents, prompt, num_opt_steps=15, device=device)
    import ipdb; ipdb.set_trace()
    # reconstruct_image_from_latents(z_T, device)
    # recons_image = reconstruct_image_from_latents(all_latents, device)

    recon_img = reconstruct(pipe, z_T, prompt, all_null_texts, guidance_scale=1, device=device)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].imshow(pp_image[0].permute(1, 2, 0).to(dtype=torch.float32).cpu().numpy())
    ax[0].set_title("Original", fontdict={'fontsize': 40})
    ax[0].axis('off')
    import ipdb; ipdb.set_trace()
    ax[1].imshow(recon_img[0])
    ax[1].set_title("Reconstructed", fontdict={'fontsize': 40})
    ax[1].axis('off')

    plt.show()

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
        device=None
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
    null_text_embeddings = torch.nn.Parameter(pipe.text_encoder(null_text_input.input_ids.to(pipe.device))[0],
                                              requires_grad=True)
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
    for timestep, prev_latents in pipe.progress_bar(zip(pipe.scheduler.timesteps, reversed(all_latents[:-1]))):
        prev_latents = prev_latents.to(pipe.device).detach()

        # expand the latents if we are doing classifier free guidance
        latent_model_input = pipe.scheduler.scale_model_input(latents, timestep).detach()
        noise_pred_text = pipe.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample.detach()
        for _ in range(num_opt_steps):
            # predict the noise residual
            noise_pred_uncond = pipe.unet(latent_model_input, timestep,
                                          encoder_hidden_states=null_text_embeddings).sample

            # perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            prev_latents_pred = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs).prev_sample
            loss = torch.nn.functional.mse_loss(prev_latents_pred, prev_latents).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        all_null_texts.append(null_text_embeddings.detach().cpu())
        latents = prev_latents_pred.detach()
    return latents, all_null_texts




def main(model_path, prompt, guidance=6.0, n_steps=20, generator=None, seed=42):
    if generator is None:
        generator = torch.manual_seed(seed)

    device = torch.device("cuda")
    model_base = "CompVis/stable-diffusion-v1-4"
    pipe = CustomDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipe.scheduler = DDIMScheduler.from_config(model_base, subfolder="scheduler")
    image1 = pipe(prompt, num_inference_steps=n_steps, guidance_scale=guidance, eta=1., generator=generator).images[0]
    import ipdb; ipdb.set_trace()
    pipe.load_model(model_path)
    pipe.to(device)
    image2 = pipe(prompt, num_inference_steps=n_steps, guidance_scale=guidance, eta=1., generator=generator).images[0]

    optimize(image1, pipe, device, prompt)

if __name__ == '__main__':
    model_path = "../concept-ablation-space/models/r2d2_delta.bin"
    prompt = "R2D2"
    main(model_path, prompt)
    # import ipdb;
    #
    # ipdb.set_trace()
    # a = 1