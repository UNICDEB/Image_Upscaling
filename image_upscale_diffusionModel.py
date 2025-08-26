import os
import torch
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline

# Input/output folders
input_folder = r"E:\Project_Work\2025\Saffron_Project\Dateset\Saffron_Flower_Data\Top_Side_Combine_Dataset_FinalCombine\Online_Image\Side_View"
output_folder = r"E:\Project_Work\2025\Saffron_Project\Dateset\Saffron_Flower_Data\Top_Side_Combine_Dataset_FinalCombine\Online_Image\upscaled_images"
os.makedirs(output_folder, exist_ok=True)

# Load Stable Diffusion Upscale pipeline
pipe = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    torch_dtype=torch.float16
).to("cuda")  # change to "cpu" if no GPU

# Loop through all images in folder
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        try:
            img_path = os.path.join(input_folder, file_name)

            # Open image
            image = Image.open(img_path).convert("RGB")

            # Upscale (empty prompt = no style change, just upscale)
            upscaled_image = pipe(prompt="", image=image).images[0]

            # Save result
            output_path = os.path.join(output_folder, file_name)
            upscaled_image.save(output_path)

            print(f"✅ Upscaled: {file_name} -> {output_path}")
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")

print("\n🎉 All images processed and saved in:", output_folder)
