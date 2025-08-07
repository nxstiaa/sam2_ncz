import torch
import torchvision 
import os 
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
from PIL import Image
from PIL import Image as PILImage
from tqdm import tqdm

#######################SET UP ###########################
# Selecting the best device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
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

'''
#Defining my writer 
log_path = os.path.abspath("outputs/logs/run1")
os.makedirs(log_path, exist_ok=True)
tracking_address = log_path #the path of my log file
writer = SummaryWriter(log_dir=log_path) 
'''


################## PART 1: ADDING BOUNDING BOXES AS INPUT #######################
'''
for each row of bounding box.csv
for filename with more than one frame, pick the earliest frame occurence
then using the filename column before.avi, look cor the corresponding directory name in /vol/bitbucket/nc624/sam2/outputs/videoFrame
on the earliest frame occurence, input the bounding box coordinate and propagate 
'''
# Paths
bbox_csv_path = "/vol/bitbucket/nc624/echonet/dynamic/bounding_boxes.csv"
video_frame_root = "/vol/bitbucket/nc624/sam2/outputs/videoFrame"
filesWithGroundtruth_csv = "/vol/bitbucket/nc624/sam2/outputs/filesWithGroundtruth.csv"

#Create saving directory
segmented_mask_dir = "/vol/bitbucket/nc624/sam2/outputs/segmentedMasksFrames2"
os.makedirs(segmented_mask_dir, exist_ok=True)
print(f"Created/verified directory: {segmented_mask_dir}")


# Load bounding box data
bb_df = pd.read_csv(bbox_csv_path)

#csv path
mask_coordinates_path = "/vol/bitbucket/nc624/echonet/dynamic/mask_coordinates.csv"
mask_df = pd.read_csv(mask_coordinates_path)

# Sort to get earliest frame
bb_df = bb_df.sort_values("Frame")
grouped = bb_df.groupby("Filename", as_index=False).first()  # Get earliest frame per video

existing_filenames = []
count = 0
video_count = 0  # Counter for processed videos

for idx, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Processing videos"): #loop through each row
    # Limit to only 2 videos
    if video_count >= 4:
        print(f"Reached limit of 2 videos. Stopping processing.")
        break 
    filename = row["Filename"]  
    ann_frame_idx = row["Frame"]
    box = np.array([row["Left_X"], row["Top_Y"], row["Right_X"], row["Bottom_Y"]], dtype=np.float32)

    # Get directory name by removing `.avi`
    video_dir_name = filename.replace(".avi", "")
    video_dir = os.path.join(video_frame_root, video_dir_name)

    # Check if directory exists
    if not os.path.isdir(video_dir):
        print(f"Skipping {video_dir_name}: directory not found")
        continue

    existing_filenames.append(filename)

    inference_state = predictor.init_state(video_path=video_dir)
    '''to reset the inference state, clears any previous prompts or segmentation'''
    # predictor.reset_state(inference_state)

    # Load frame names
    frame_names = sorted([
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"] #filters for only jpg images 
    ])

    if ann_frame_idx >= len(frame_names):
        print(f"Frame index {ann_frame_idx} out of range for {video_dir_name}")
        continue

    print(f"Processing: {filename} at frame {ann_frame_idx}")

    # Load the frame image for visualization
    frame_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    frame_img = Image.open(frame_path).convert("RGB")
    frame_np = np.array(frame_img)

    # 1. Save image with bounding box
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame_np)
    show_box(box, ax)
    ax.set_title(f"Image with Bounding Box - {filename} Frame {ann_frame_idx}")
    ax.axis('off')
    bbox_save_path = os.path.join(segmented_mask_dir, f"{video_dir_name}_frame{ann_frame_idx}_bbox.png")
    plt.savefig(bbox_save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved bounding box image: {bbox_save_path}")

    ann_obj_id = 1  # assuming this is set somewhere in your environment
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box,
    )

    # 2. Save mask after initial bounding box prompt
    initial_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame_np)
    show_mask(initial_mask, ax)
    show_box(box, ax)
    ax.set_title(f"Mask after Bounding Box Prompt - {filename} Frame {ann_frame_idx}")
    ax.axis('off')
    initial_mask_save_path = os.path.join(segmented_mask_dir, f"{video_dir_name}_frame{ann_frame_idx}_initial_mask.png")
    plt.savefig(initial_mask_save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved initial mask: {initial_mask_save_path}")

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
    # for labels, `1` means positive click and `0` means negative click
    labels = np.ones(len(points), dtype=np.int32)
    # note that we also need to send the original box input along with
    # the new refinement click together into `add_new_points_or_box`
    box = np.array([row["Left_X"], row["Top_Y"], row["Right_X"], row["Bottom_Y"]], dtype=np.float32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
        box=box,
    )

    # 3. Save mask after refinement prompts
    refined_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame_np)
    show_mask(refined_mask, ax)
    show_box(box, ax)
    show_points(points, labels, ax)  # Show the refinement points
    ax.set_title(f"Mask after Refinement Prompts - {current_filename} Frame {ann_frame_idx}")
    ax.axis('off')
    refined_mask_save_path = os.path.join(segmented_mask_dir, f"{video_dir_name}_frame{ann_frame_idx}_refined_mask.png")
    plt.savefig(refined_mask_save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved refined mask: {refined_mask_save_path}")

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    #Get the list of annotated frames for this video
    framesToSave = bb_df[bb_df["Filename"] == filename]["Frame"].tolist()

    for frame in framesToSave:
        if frame in video_segments and ann_obj_id in video_segments[frame]:
                #create a white background 112x112
                background = np.full((112,112,3), 255, dtype=np.uint8)

                mask_array = video_segments[frame][ann_obj_id]
                # Ensure mask_array is 2D before resizing
                if mask_array.ndim > 2:
                    mask_array = mask_array.squeeze()  # Remove extra dimensions
                mask_resized = cv2.resize(mask_array.astype(np.uint8), (112, 112), interpolation=cv2.INTER_NEAREST)


                # A 4-channel image has shape (height, width, 4)
                # The 4 channels are:
                # Index 0: Red channel
                # Index 1: Green channel  
                # Index 2: Blue channel
                # Index 3: Alpha channel (transparency)
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
                print(f"Saved mask for {filename} at frame {frame}")
    
    video_count += 1  # Increment video counter after processing each video
    print(f"Completed processing video {video_count}/2: {filename}")
    
print(f"No. of files saved = {count}")
print(f"Total videos processed: {video_count}")