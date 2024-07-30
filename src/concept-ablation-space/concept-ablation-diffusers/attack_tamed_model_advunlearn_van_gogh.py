from diffusers import StableDiffusionPipeline, DDIMScheduler
from model_pipeline import CustomDiffusionPipeline
from transformers import CLIPTextModel
import torch
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from matplotlib import pyplot as plt

has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')

model_id_or_path = "CompVis/stable-diffusion-v1-4"
scheduler = DDIMScheduler.from_config(model_id_or_path, subfolder="scheduler")
pipe = CustomDiffusionPipeline.from_pretrained(
    model_id_or_path,
    safety_checker=None,
    scheduler = scheduler,
).to(device)

prompt = "The Starry Night"
image1 = pipe(prompt, num_inference_steps=50, guidance_scale=6.0, eta=1., generator=None).images[0]
image_path = f"attack_input_{prompt.replace(' ', '')}.png"
image1.save(image_path)
SRC_IMAGE = image_path
RESULT_IMAGE_PATH = f"attack_output_{prompt.replace(' ', '')}_with_an_edit_red_skies.png"
# load advunlearn result:
cache_path = 'concept-ablation-diffusers/cache_dir/'
text_encoder = CLIPTextModel.from_pretrained("OPTML-Group/AdvUnlearn", subfolder="vangogh_unlearned", cache_dir=cache_path)
pipe.text_encoder = text_encoder

pipe.to(device)

image = Image.open(SRC_IMAGE).convert("RGB")
image = image.resize((512, 512))

def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image

pp_image = preprocess_image(image)

generator = torch.Generator(device=device)
latent = pipe.vae.encode(pp_image.to(device)).latent_dist.sample(generator=generator) *  0.18215

input_prompt = "The Starry Night red skies"
text_embeddings = pipe._encode_prompt(input_prompt, device, 1, False, None)


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


num_inference_steps = 50
pipe.scheduler.set_timesteps(num_inference_steps, device=device)
all_latents = ddim_inversion(latent, text_embeddings, pipe.scheduler, pipe.unet)


def null_text_inversion(
        pipe,
        all_latents,
        prompt,
        num_opt_steps=15,
        lr=0.01,
        tol=1e-5,
        guidance_scale=7.5,
        eta: float = 0.0,
        generator=None
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
    return all_latents[-1], all_null_texts


z_T, all_null_texts = null_text_inversion(pipe, all_latents, input_prompt, num_opt_steps=15)


@torch.no_grad()
def reconstruct(pipe, latents, prompt, null_text_embeddings, guidance_scale=7.5, generator=None, eta=0.0):
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

    #Post-processing
    image = pipe.decode_latents(latents)
    return image


recon_img = reconstruct(pipe, z_T, input_prompt, all_null_texts, guidance_scale=1)
def plot_me(pp_image, recon_img):
    plt.close('all')
    plt.subplot(1, 3, 1)
    plt.imshow(pp_image[0].permute(1,2,0).numpy() * 0.5 + 0.5)
    plt.title("Original")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(recon_img[0])
    plt.title("Reconstructed using Null Text Inversion")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(recon_img[0] - (pp_image[0].permute(1,2,0).numpy() * 0.5 + 0.5)).mean(axis=-1) * 255, cmap='grey')
    plt.colorbar()
    plt.title("Gray Scale Diff \nbetween original and Reconstructed")
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches((12, 4))
    plt.tight_layout()
    plt.savefig(RESULT_IMAGE_PATH)

plot_me(pp_image, recon_img)

