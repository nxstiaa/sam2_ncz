import torch
import torchvision 
import os
import os, time
import io
import subprocess
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import zipfile 
import cv2
import shutil
import subprocess
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from PIL import Image as PILImage
from tqdm import tqdm
from model import build_unet
from utils import create_dir, seeding 
from train import load_data 
#from Test_files.augmentation import main 

#######################SET UP ###########################
# Selecting the best device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Set memory management environment variable
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Clear GPU cache at start
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Enable precision tuning for CUDA
if device.type == "cuda":
    # Enable TF32 for Ampere GPUs (compute capability >= 8)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

elif device.type == "mps":
    print(
        "\n[Warning] Support for MPS devices is preliminary. SAM 2 was trained with CUDA and may "
        "produce numerically different outputs or slower performance on MPS."
        "\nSee: https://github.com/pytorch/pytorch/issues/84936"
    )


########################### ENV SETUP ################################
# display system info
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

'''
# Install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python", "matplotlib"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/sam2.git"])

# Ensuring the folders exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs/VidWGroundTruth", exist_ok=True)
'''


# Download the model checkpoint only if it doesn't exist
checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"

if not os.path.exists(sam2_checkpoint):
    print("Downloading SAM2 model checkpoint...")
    subprocess.check_call([
        "wget", "-P", "checkpoints", checkpoint_url
    ])
    print("Download completed!")
else:
    print("SAM2 model checkpoint already exists, skipping download.")

############### LOADING SAM2 VIDEO PREDICTOR ####################
from sam2.build_sam import build_sam2_video_predictor

model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Initialize the SAM2 video predictor only once
print("Loading SAM2 video predictor...")
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
print("SAM2 video predictor loaded successfully!")

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def log_image_to_tensorboard(writer, image_pil, tag="Image", step=0):
    image = image_pil.convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_np).permute(2, 0, 1)
    writer.add_image(tag, image_tensor, global_step=step)

def print_gpu_memory():
    """Print current GPU memory usage"""
    if device.type == "cuda":
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'total    : {info.total/1024**3} GB')
        print(f'free     : {info.free/1024**3} GB')
        print(f'used     : {info.used/1024**3} GB')

#Defining my writer 
# log_path = os.path.abspath("outputs/logs/run1")
# os.makedirs(log_path, exist_ok=True)
# tracking_address = log_path #the path of my log file
# writer = SummaryWriter(log_dir=log_path) 

processed_video_count = 0

# Paths
bbox_csv_path = "/vol/bitbucket/nc624/echonet/dynamic/ground_truth/bounding_boxes.csv"
video_frame_root = "/vol/bitbucket/nc624/echonet/img_preprocessing/preprocessed_frames"

#Create saving directory
segmented_mask_dir = "/vol/bitbucket/nc624/sam2/outputs/segmentedMasksFrames_PreProcessed"
os.makedirs(segmented_mask_dir, exist_ok=True)
print(f"Created/verified directory: {segmented_mask_dir}")

######################### PART 0: AUGMENTATION #########################
def horizontal_flip(image: np.ndarray) -> np.ndarray:
    return np.flip(image, axis=1)

def horizontal_coordinate_flip(coordinates: np.ndarray, image_width: int) -> np.ndarray:
    flipped_coords = coordinates.copy()
    flipped_coords[:, 0] = image_width - flipped_coords[:, 0]
    return flipped_coords


def scale_image(image: np.ndarray, scale_factor: float):
    h, w = image.shape[:2]
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    # Resize to new scaled size
    scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Then center-crop or pad back to (112, 112)
    return cv2.resize(scaled_image, (112, 112), interpolation=cv2.INTER_LINEAR)

def transform_coords(x, y, original_size, scale_factor, final_size=(112, 112)):
    orig_h, orig_w = original_size
    new_h = int(orig_h * scale_factor)
    new_w = int(orig_w * scale_factor)

    # Step 1: Scale the coordinates
    x_scaled = x * scale_factor
    y_scaled = y * scale_factor

    # Step 2: Resize into 112x112 space
    # Coordinates must also be scaled according to how the image was resized to 112x112
    scale_x = final_size[1] / new_w
    scale_y = final_size[0] / new_h

    x_final = x_scaled * scale_x
    y_final = y_scaled * scale_y

    return x_final, y_final

    
def gaussian_blur(image: np.ndarray, sigma_range: tuple = (0.1, 0.8)) -> np.ndarray:
    # Very light blur only - don't add to existing noise
    sigma = np.random.uniform(*sigma_range)
    return cv2.GaussianBlur(image, (0, 0), sigma)

def adjust_brightness(image: np.ndarray, factor_range: tuple = (0.6, 1.4)) -> np.ndarray:
    # Even wider range - noise makes gain adjustments more critical
    factor = np.random.uniform(*factor_range)
    return np.clip(image * factor, 0, 255).astype(image.dtype)

def adjust_contrast(image: np.ndarray, factor_range=(0.75, 1.25)):
    factor = np.random.uniform(*factor_range)
    mean = image.mean()
    return np.clip((image - mean) * factor + mean, 0, 255).astype(image.dtype)


################## PART 1: ADDING BOUNDING BOXES AS INPUT #######################
'''
for each row of bounding box.csv
for filename with more than one frame, pick the earliest frame occurence
then using the filename column before.avi, look cor the corresponding directory name in /vol/bitbucket/nc624/sam2/outputs/videoFrame
on the earliest frame occurence, input the bounding box coordinate and propagate 
'''

# Load bounding box data
bb_df = pd.read_csv(bbox_csv_path)

#csv path
mask_coordinates_path = "/vol/bitbucket/nc624/echonet/dynamic/ground_truth/groundTruthMask_coordinates.csv"
mask_df = pd.read_csv(mask_coordinates_path)

# Sort to get earliest frame
bb_df = bb_df.sort_values("Frame")
grouped_first = bb_df.groupby("Filename", as_index=False).first()  # Get earliest frame per video
grouped_last = bb_df.groupby("Filename", as_index=False).last()    # Get latest frame per video

existing_filenames = []
count = 0
video_count = 0  # Counter for processed videos
visualisation_count = 0

for idx, row in tqdm(grouped_first.iterrows(), total=len(grouped_first), desc="Processing videos"): #loop through each row 
    filename = row["Filename"]  # e.g., "0X18B2F3A2E992AF3E.avi"
    ann_frame_idx = row["Frame"]
    box = np.array([row["Left_X"], row["Top_Y"], row["Right_X"], row["Bottom_Y"]], dtype=np.float32)
    tta_box = aug(box)

    # Get directory name by removing `.avi`
    video_dir_name = filename.replace(".avi", "")
    video_dir = os.path.join(video_frame_root, video_dir_name)

    # Check if directory exists
    if not os.path.isdir(video_dir):
        print(f"Skipping {video_dir_name}: directory not found")
        continue

    # Check if segmented masks already exist for this video
    # Look for files that start with the video name pattern
    existing_mask_files = [f for f in os.listdir(segmented_mask_dir) 
                          if f.startswith(video_dir_name + "_frame") and f.endswith('.png')]
    
    if existing_mask_files:
        print(f"Skipping {video_dir_name}: segmented masks already exist ({len(existing_mask_files)} files found)")
        continue

    existing_filenames.append(filename)

    inference_state = predictor.init_state(video_path=video_dir)
    '''to reset the inference state, clears any previous prompts or segmentation'''
    # predictor.reset_state(inference_state)
    
    # Clear memory after initializing inference state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Load frame names
    frame_names = sorted([
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"] #filters for only jpg images 
    ])

    if ann_frame_idx >= len(frame_names):
        print(f"Frame index {ann_frame_idx} out of range for {video_dir_name}")
        continue

    total_frames = len(frame_names)
    
    # Get the last frame with annotations for this filename
    last_annotation_row = grouped_last[grouped_last["Filename"] == filename].iloc[0] #Get the last frame of the specific filename
    last_annotation_frame = last_annotation_row["Frame"]
    
    print(f"Processing: {filename} at frame {ann_frame_idx} (total frames: {total_frames}, last annotation frame: {last_annotation_frame})")

    # Load the actual image for augmentation
    frame_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    #Load array as a numpy array 
    frame_image = cv2.imread(frame_path)
    #Convert from BGR to RGB
    frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
    
    augmented_img = aug(frame_image)

    
    ann_obj_id = 1  # assuming this is set somewhere in your environment
    with torch.no_grad():
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box,
        )

    ################## PART 2: ADDING REFINEMENT PROMPTS #########################
    '''
    now for each filename with ground truth in the csv file 
    we want 3 randomly generated coordinates of the ground truth mask from the same first frame
    and then the respective box coordinate
    propagate it through 
    save only the 2 frame numbers 
    '''

    #Get all rows for with that filename 
    current_filename = filename
    filename_rows = mask_df[(mask_df["FileName"] == current_filename) & (mask_df["Frame"] == ann_frame_idx)]


    if len(filename_rows) >= 3:
        #Randomly select 3 rows from the dataframe
        sampled_rows = filename_rows.sample(n=3, random_state=42)
    else:
        sampled_rows = filename_rows

    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    #add a positive click 
    points = sampled_rows[['X', 'Y']].values.astype(np.float32)
    tta_points = aug(points)

    # for labels, `1` means positive click and `0` means negative click
    labels = np.ones(len(points), dtype=np.int32)
    # note that we also need to send the original box input along with
    # the new refinement click together into `add_new_points_or_box`
    #box = np.array([row["Left_X"], row["Top_Y"], row["Right_X"], row["Bottom_Y"]], dtype=np.float32)

    with torch.no_grad():
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=tta_points,
            labels=labels,
            box=box,
        )

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    
    # Propagate through the video until the last frame with annotations
    print(f"Starting propagation from frame {ann_frame_idx} to frame {last_annotation_frame}")
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # Stop propagation if we've reached the last annotation frame
        if out_frame_idx > last_annotation_frame:
            print(f"Completed propagation at frame {out_frame_idx-1} (last annotation frame: {last_annotation_frame})")
            break
            
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    # Clear memory after video propagation
    if device.type == "cuda":
        torch.cuda.empty_cache()

    #Get the list of annotated frames for this video
    framesToSave = bb_df[bb_df["Filename"] == current_filename]["Frame"].tolist()

    for frame in framesToSave:
        if frame in video_segments and ann_obj_id in video_segments[frame]:
                #create a white background 112x112
                background = np.full((112,112,3), 255, dtype=np.uint8)

                mask_array = video_segments[frame][ann_obj_id]
                # Ensure mask_array is 2D before resizing
                if mask_array.ndim > 2:
                    mask_array = mask_array.squeeze()  # Remove extra dimensions
                mask_resized = cv2.resize(mask_array.astype(np.uint8), (112, 112), interpolation=cv2.INTER_NEAREST)

                colored_mask = np.zeros((112,112,4), dtype=np.uint8) #initialise blank image
                colored_mask[:, :, 0] = 255 #make everything red
                colored_mask[:, :, 3] = mask_resized * 255 # anything that is 0 is transparent

                # Convert background and mask to PIL 
                white_bg = Image.fromarray(background).convert("RGBA")
                color_mask = Image.fromarray(colored_mask)

                #Overlay red mask on white background
                blended = Image.alpha_composite(white_bg, color_mask)

                # Save the result
                save_path = os.path.join(segmented_mask_dir, f"{video_dir_name}_frame{frame}.png")
                blended.save(save_path)
                count += 1
                print(f"Saved mask for {current_filename} at frame {frame}")
    
    # Clear GPU memory after processing each video
    if device.type == "cuda":
        torch.cuda.empty_cache()  # Clear PyTorch's cached memory
        print_gpu_memory()
    
    video_count += 1  # Increment video counter after processing each video
    print(f"Completed processing video {video_count}: {current_filename}")
    
print(f"No. of files saved = {count}")
print(f"Total videos processed: {video_count}")
