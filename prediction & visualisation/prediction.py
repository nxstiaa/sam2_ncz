import cv2
import os
import pandas as pd
import numpy as np
from medpy.metric.binary import hd95
from collections import defaultdict

'''
go through the segmentedMasksFrames directory 
Get the filename and frame number 
find the corresponding ground truth MASK IMAGE

#Calculate the DICE score and hausdorff 95 for each of these frames, output to a csv file
#from the csv file, calculate the mean and standard deviation of DICE and Hausdorff 95
'''

# Get sorted lists of files from both directories
ground_truth_dir = "/vol/bitbucket/nc624/echonet/dynamic/ground_truth/groundTruthMasks"
segmented_mask_dir = "/vol/bitbucket/nc624/sam2/outputs/segmentedMasksFrames_PreProcessed"

######Helper functions & Getting Binary Mask ######
def extract_red_mask(img):
    red = (img[:, :, 2] > 150)
    green = (img[:, :, 1] < 50)
    blue = (img[:, :, 0] < 50)
    return (red & green & blue).astype(np.uint8)

def extract_blue_mask(img):
    blue = (img[:, :, 0] >150)
    green = (img[:, :, 1] <50)
    red = (img[:, :, 2] <50)
    return (blue & green & red).astype(np.uint8) # T/F becomes 1/0

mask_files = [f for f in os.listdir(segmented_mask_dir) if f.endswith('.png')] #list all png files 
grouped = defaultdict(list) #grouped is a dictionary where the key is FileName and value is a list of frames


for file in mask_files:
    base_name = file.split("_frame")[0]
    frame_number = int(file.split("_frame")[1].split(".")[0])
    grouped[base_name].append((frame_number, file)) #add the frame number to the dictionary, it auto-handles same base names


results = []

for video_name, frames in grouped.items(): #return the key value pairs
    if len(frames) < 2:
        print(f"Skipping {video_name}, expected 2 frames, got {len(frames)}")
        continue

    frames = sorted(frames, key = lambda x: x[0]) #use frame no. for sorting, sort by each element in the tuple
    (diastole_frame, diastole_filename), (systole_frame, systole_filename) = frames #assign the first and second elements of the tuple to diastole and systole


    #get the ground truth mask
    diastole_gt_filename = f"{video_name}_{diastole_frame}.png"
    systole_gt_filename = f"{video_name}_{systole_frame}.png"

    diastole_gt_filepath = os.path.join(ground_truth_dir, diastole_gt_filename)
    systole_gt_filepath = os.path.join(ground_truth_dir, systole_gt_filename)

    if not os.path.exists(diastole_gt_filepath):
        print("Diastole ground truth mask not found")
        continue

    if not os.path.exists(systole_gt_filepath):
        print("Systole ground truth mask not found")
        continue

    #Read the ground truth mask and segmented mask
    diastole_gt = cv2.imread(diastole_gt_filepath)
    systole_gt = cv2.imread(systole_gt_filepath)
    diastole_seg = cv2.imread(os.path.join(segmented_mask_dir, diastole_filename))
    systole_seg = cv2.imread(os.path.join(segmented_mask_dir, systole_filename))

    # Check if images loaded successfully
    if diastole_seg is None or systole_seg is None or diastole_gt is None or systole_gt is None:
        print(f"Skipping {video_name}: one or more images couldn't be read.")
        continue

    ########## Getting Binary Masks #########
    binary_diastole_gt = extract_blue_mask(diastole_gt)
    binary_systole_gt = extract_blue_mask(systole_gt)
    binary_diastole_segmentedMask = extract_red_mask(diastole_seg)
    binary_systole_segmentedMask = extract_red_mask(systole_seg)
     
    if binary_diastole_gt.shape != binary_diastole_segmentedMask.shape:
        print(f"Skipping {video_name}: shape mismatch.")
        continue

    ########## DICE CALCULATION ##########
    #diastole 
    diastole_intersection = np.logical_and(binary_diastole_gt, binary_diastole_segmentedMask).sum()
    diastole_union = np.logical_or(binary_diastole_gt, binary_diastole_segmentedMask).sum()
    large_dice = 2 * diastole_intersection / (diastole_union + diastole_intersection + 1e-8)

    #systole
    systole_intersection = np.logical_and(binary_systole_gt,  binary_systole_segmentedMask).sum()
    systole_union = np.logical_or(binary_systole_gt,  binary_systole_segmentedMask).sum()
    small_dice = 2 * systole_intersection / (systole_union + systole_intersection + 1e-8)

    #overall
    overall_dice = 2 * (diastole_intersection + systole_intersection) / (diastole_union + diastole_intersection + systole_union + systole_intersection + 1e-8)

    ########## HAUSDOFF CALCULATION ##########
    diastole_haus = hd95(binary_diastole_segmentedMask, binary_diastole_gt)
    systole_haus = hd95(binary_systole_segmentedMask, binary_systole_gt)
    overall_haus = (hd95(binary_diastole_segmentedMask, binary_diastole_gt) + hd95(binary_systole_segmentedMask, binary_systole_gt))/2

    results.append({
        'FileName': video_name,
        "DiastoleDICE": large_dice,
        "SystoleDICE": small_dice,
        "OverallDICE": overall_dice,
        "OverallHausdorff95": overall_haus,
        "DiastoleHausdorff95": diastole_haus,
        "SystoleHausdorff95": systole_haus
    })
    

#write results to csv file
results_df = pd.DataFrame(results)
results_df.to_csv("/vol/bitbucket/nc624/sam2/CSV_Files/performance_preprocessed.csv", index=False)
print("DICE and Hausdorff 95 scores saved.")


########## Summary statistics ##########
mean_diastole_dice = results_df["DiastoleDICE"].mean()
std_diastole_dice = results_df["DiastoleDICE"].std()

mean_systole_dice = results_df["SystoleDICE"].mean()
std_systole_dice = results_df["SystoleDICE"].std()

mean_overall_dice = results_df["OverallDICE"].mean()
std_overall_dice = results_df["OverallDICE"].std()

mean_diastole_haus = results_df["DiastoleHausdorff95"].mean()
std_diastole_haus = results_df["DiastoleHausdorff95"].std()

mean_systole_haus = results_df["SystoleHausdorff95"].mean()
std_systole_haus = results_df["SystoleHausdorff95"].std()

mean_overall_haus = results_df["OverallHausdorff95"].mean()
std_overall_haus = results_df["OverallHausdorff95"].std()

summary_stats = {
    "Metric": [
        "DICE", "DICE", "DICE", 
        "Hausdorff95", "Hausdorff95", "Hausdorff95"
    ],
    "Phase": [
        "Diastole", "Systole", "Overall", 
        "Diastole", "Systole", "Overall"
    ],
    "Mean": [
        mean_diastole_dice, mean_systole_dice, mean_overall_dice,
        mean_diastole_haus, mean_systole_haus, mean_overall_haus
    ],
    "Std": [
        std_diastole_dice, std_systole_dice, std_overall_dice,
        std_diastole_haus, std_systole_haus, std_overall_haus
    ]
}
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv("/vol/bitbucket/nc624/sam2/CSV_Files/PerformanceSummary_preprocessed.csv", index=False)

print("Performance log completed")

