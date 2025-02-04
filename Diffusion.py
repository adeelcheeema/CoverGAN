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
    generator = torch.Generator(device="cude").manual_seed(0)
    images = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=7.5,
        generator=generator,
        num_images_per_prompt=3,
    ).images

    images
    return images



# image_grid(images, 1, num_samples + 1)