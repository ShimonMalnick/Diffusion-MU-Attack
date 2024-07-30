import gradio as gr
import PIL.Image
import shlex
import shutil
import subprocess
from pathlib import Path
import os
import torch
from tqdm import tqdm



def pad_image(image: PIL.Image.Image) -> PIL.Image.Image:
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = PIL.Image.new(image.mode, (w, w), (0, 0, 0))
        new_image.paste(image, (0, (w - h) // 2))
        return new_image
    else:
        new_image = PIL.Image.new(image.mode, (h, h), (0, 0, 0))
        new_image.paste(image, ((h - w) // 2, 0))
        return new_image



def train_submit(
        prompt, anchor_prompt, concept_type, reg_lambda, iterations, lr, openai_key, save_path, mem_impath=None
    ):
        if not torch.cuda.is_available():
            raise gr.Error('CUDA is not available.')

        torch.cuda.empty_cache()
        original_prompt = prompt
        parameter_group = "cross-attn"
        train_batch_size = 4
        if concept_type == 'style':
            class_data_dir = f'./data/samples_painting/'
            anchor_prompt = f'./assets/painting.txt'
            openai_key = ''
        elif concept_type == 'object':
            os.makedirs('temp', exist_ok=True)
            class_data_dir = f'./temp/{anchor_prompt}'
            name = save_path.split('/')[-1]
            prompt = f'{anchor_prompt}+{prompt}'
            assert openai_key is not None

            if len(openai_key.split('\n')) > 1:
                openai_key = openai_key.split('\n')
                with open(f'./temp/{name}.txt', 'w') as f:
                    for prompt_ in openai_key:
                        f.write(prompt_.strip()+'\n')
                openai_key = ''
                anchor_prompt = f'./temp/{name}.txt'
        elif concept_type == 'memorization':
            os.system("wget https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_mixup.torchscript.pt -P assets/") 
            os.makedirs('temp', exist_ok=True)
            prompt = f'*+{prompt}'
            name = save_path.split('/')[-1]
            train_batch_size = 1
            lr = 5e-7
            parameter_group = "full-weight"

            assert openai_key is not None
            assert mem_impath is not None

            if len(openai_key.split('\n')) > 1:
                openai_key = openai_key.split('\n')
                with open(f'./temp/{name}.txt', 'w') as f:
                    for prompt_ in openai_key:
                        f.write(prompt_.strip()+'\n')
                openai_key = ''
                anchor_prompt = f'./temp/{name}.txt'
            else:
                anchor_prompt = prompt

            print(mem_impath)
            image = PIL.Image.open(mem_impath[0][0].name)
            image = pad_image(image)
            image = image.convert('RGB')
            mem_impath = f"./temp/{original_prompt.lower().replace(' ', '')}.jpg"
            image.save(mem_impath, format='JPEG', quality=100)

            class_data_dir = f"./temp/{original_prompt.lower().replace(' ', '')}"
            

        command = f'''
        accelerate launch concept-ablation-diffusers/train.py \
          --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
          --output_dir={save_path} \
          --class_data_dir={class_data_dir} \
          --class_prompt="{anchor_prompt}"  \
          --caption_target "{prompt}" \
          --concept_type {concept_type} \
          --resolution=512  \
          --train_batch_size={train_batch_size}  \
          --learning_rate={lr}  \
          --max_train_steps={iterations} \
          --scale_lr --hflip \
          --parameter_group {parameter_group} \
          --openai_key "{openai_key}" \
          --enable_xformers_memory_efficient_attention  --num_class_images 500
        '''

        if concept_type == 'style':
            command += f'  --noaug'
        
        if concept_type == 'memorization':
            command += f' --use_8bit_adam  --with_prior_preservation --prior_loss_weight=1.0 --mem_impath {mem_impath}'
            
        with open(f'{save_path}/train.sh', 'w') as f:
            command_s = ' '.join(command.split())
            f.write(command_s)

        res = subprocess.run(shlex.split(command))

        if res.returncode == 0:
            result_message = 'Training Completed!'
        else:
            result_message = 'Training Failed!'
        weight_paths = sorted(Path(save_path).glob('*.bin'))
        print(weight_paths)
        return gr.update(value=result_message), weight_paths[0]


def inference(model_path, prompt, guidance, n_steps, generator):
    import sys 
    sys.path.append('concept-ablation/diffusers/.')
    from model_pipeline import CustomDiffusionPipeline
    import torch

    pipe = CustomDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
    image1 = pipe(prompt, num_inference_steps=n_steps, guidance_scale=guidance, eta=1., generator=generator).images[0]

    pipe.load_model(model_path)
    image2 = pipe(prompt, num_inference_steps=n_steps, guidance_scale=guidance, eta=1., generator=generator).images[0]

    return image1, image2