'''
Checking which coordinates in the combined_dataFrame.csv 
making sure between rows, the coordinates are not more than 10 pixels apart
'''

import pandas as pd

# Load the list of test videos
available_videos_path = "/vol/bitbucket/nc624/echonet/dynamic/available_videos.txt"
with open(available_videos_path, "r") as f:
    test_videos = [line.strip() for line in f.readlines()]

print(f"Loaded {len(test_videos)} test videos from available_videos.txt")

df = pd.read_csv("/vol/bitbucket/nc624/echonet/dynamic/ground_truth/combined_dataFrame.csv")

#multiple columns need brackets
#sorting by filename and frame, creating a single dataframe
df = df.sort_values(by=["FileName", "Frame"]) 

# Filter the dataframe to only include test videos
df = df[df["FileName"].isin(test_videos)]
print(f"Filtered to {len(df)} rows from test videos")

flagged_filenames = []

#group is a DataFrame containing all rows where FileName == filename and Frame == frame.
#splits the dataframe into smaller dataframes 
for filename, group in df.groupby("FileName"):
    group = group.sort_values(by="Frame")

    for i in range(len(group) - 1):
        current_row = group.iloc[i] #select row by integer postiton
        next_row = group.iloc[i + 1]

        # Only compare coordinates if they are from the same frame
        if current_row["Frame"] == next_row["Frame"]:
            current_x = current_row["X"]
            current_y = current_row["Y"]
            next_x = next_row["X"]
            next_y = next_row["Y"]

            difference_x = abs(next_x - current_x)
            difference_y = abs(next_y - current_y)

            if difference_x > 30 or difference_y > 30:
                flagged_filenames.append({
                    "FileName": filename,
                    "Frame_current": current_row["Frame"],
                    "X_current": current_x,
                    "Y_current": current_y,
                    "Frame_next": next_row["Frame"],
                    "X_next": next_x,
                    "Y_next": next_y,
                    "Diff_X": difference_x,
                    "Diff_Y": difference_y
                })

#first convert set to a dataframe, because set does not hv a .csv method 
flagged_filenames = pd.DataFrame(flagged_filenames)
#then convert it to a csv file
flagged_filenames.to_csv("/vol/bitbucket/nc624/sam2/Test_files/flagged_filenames.csv", index=False)
        


