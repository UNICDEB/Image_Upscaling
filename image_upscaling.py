import os
from PIL import Image
from realesrgan import RealESRGAN

# Input and output folders
input_folder = "E:\Project_Work\2025\Saffron_Project\Dateset\Saffron_Flower_Data\Top_Side_Combine_Dataset_FinalCombine\Online_Image\Side_View"
output_folder = "E:\Project_Work\2025\Saffron_Project\Dateset\Saffron_Flower_Data\Top_Side_Combine_Dataset_FinalCombine\Online_Image\upscaled_images"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Load Real-ESRGAN model
model = RealESRGAN.from_pretrained('RealESRGAN_x4plus')
print("✅ Model loaded successfully!")

# Loop through all images in input folder
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        img_path = os.path.join(input_folder, file_name)
        try:
            # Open image
            img = Image.open(img_path).convert("RGB")

            # Upscale
            sr_image = model.predict(img)

            # Save result
            output_path = os.path.join(output_folder, file_name)
            sr_image.save(output_path)

            print(f"✅ Upscaled: {file_name} -> {output_path}")
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")

print("\n🎉 All images processed and saved in:", output_folder)
