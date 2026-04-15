import h5py
import numpy as np
from PIL import Image
import os


def extract_nyu_images(mat_file_path, output_dir='extracted_dataset'):
    os.makedirs(f"../../{output_dir}/images", exist_ok=True)
    os.makedirs(f"../../{output_dir}/depths", exist_ok=True)
    
    with h5py.File(mat_file_path, 'r') as f:
        images = f['images']
        depths = f['depths']
        num_total =images.shape[0]
        
        for i in range(num_total):
            img = images[i].transpose(1, 2, 0)
            
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            img_pil = Image.fromarray(img)
            img_pil.save(f"{output_dir}/images/image_{i:04d}.png")
            
            depth = depths[i]
            depth_min, depth_max = depth.min(), depth.max()
            if depth_max > depth_min:
                depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 255
            else:
                depth_normalized = np.zeros_like(depth)
            
            depth_img = Image.fromarray(depth_normalized.astype(np.uint8))
            depth_img.save(f"{output_dir}/depths/depth_{i:04d}.png")
            np.save(f"{output_dir}/depths/depth_{i:04d}.npy", depth)
            
            if (i + 1) % 20 == 0:
                print(f"Extracted {i + 1}/{num_total}")
        
        print("Done")