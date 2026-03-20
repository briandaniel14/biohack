import os
from tifffile import imread, imwrite

# Path to data folder (relative to scripts/)
input_dir = "../data"
output_dir = os.path.join(input_dir, "separated_frames")

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".tif") or file.endswith(".tiff"):
        path = os.path.join(input_dir, file)
        print(f"Processing {file}...")

        try:
            img = imread(path)
            print(f"Shape: {img.shape}, dtype: {img.dtype}")
        except Exception as e:
            print(f"Skipping {file}: {e}")
            continue

        # Time series splitting by frame
        if img.ndim == 3:
            print(f"{file} is a time series with {img.shape[0]} frames")

            for t in range(img.shape[0]):
                frame = img[t]

                out_name = f"{os.path.splitext(file)[0]}_frame_{t}.tiff"
                out_path = os.path.join(output_dir, out_name)

                imwrite(out_path, frame)

        else:
            print(f"Skipping {file}: unsupported shape {img.shape}")