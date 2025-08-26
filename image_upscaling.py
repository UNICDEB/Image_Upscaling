import os
from PIL import Image
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import numpy as np

# Input/output folders
input_folder = r"E:\Project_Work\2025\Saffron_Project\Dateset\Saffron_Flower_Data\Top_Side_Combine_Dataset_FinalCombine\Online_Image\Side_View"
output_folder = r"E:\Project_Work\2025\Saffron_Project\Dateset\Saffron_Flower_Data\Top_Side_Combine_Dataset_FinalCombine\Online_Image\upscaled_images"

os.makedirs(output_folder, exist_ok=True)

# Choose device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Load model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)

upsampler = RealESRGANer(
    scale=4,
    model_path='RealESRGAN_x4plus.pth',
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True,
    device=torch.device(device)
)

# Loop through images
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_folder, file_name)
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)

            # Upscale
            output, _ = upsampler.enhance(img_np, outscale=4)

            # Save
            output_img = Image.fromarray(output)
            output_img.save(os.path.join(output_folder, file_name))
            print(f"✅ Upscaled: {file_name}")
        except Exception as e:
            print(f"❌ Error: {file_name} -> {e}")
