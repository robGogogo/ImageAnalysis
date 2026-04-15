from dataset_extract import extract_nyu_images
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
mat_file_path = os.path.join(BASE_DIR, "training_dataset", "nyu_depth_v2_labeled.mat")

extract_nyu_images(mat_file_path)   