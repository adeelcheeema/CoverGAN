import PIL
import requests
import torch


from io import BytesIO

from diffusers import StableDiffusionInpaintPipeline
import PIL

import torch
print(torch.__version__)
print(torch.cuda.is_available())

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    variant="fp16",
    torch_dtype=torch.float16,
)

pipe = pipe.to("cuda")


def gen_image(prompt, image, mask_image):
    generator = torch.Generator(device="cuda").manual_seed(0)
    pipe.safety_checker = None
    images = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=7.5,
        generator=generator,
        strength=1,
        num_images_per_prompt=3,
    ).images

    return images



# image_grid(images, 1, num_samples + 1)