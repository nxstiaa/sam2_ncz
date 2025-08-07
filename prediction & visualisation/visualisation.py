import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

'''
# Load data from CSV files
original_data = pd.read_csv("/vol/bitbucket/nc624/sam2/CSV_Files/performance_nonProcessedSummary.csv")
preprocessed_data = pd.read_csv("/vol/bitbucket/nc624/sam2/CSV_Files/PerformanceSummary_preprocessed.csv")

# Plot settings
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
metrics = ["DICE", "Hausdorff95"]
phases = ["Diastole", "Systole", "Overall"]
x = np.arange(len(phases))
width = 0.35

for i, metric in enumerate(metrics):
    ax = axs[i]
    ori = original_data[original_data["Metric"] == metric]
    pre = preprocessed_data[preprocessed_data["Metric"] == metric]

    ax.bar(x - width/2, ori["Mean"], width, yerr=ori["Std"], label='Original', capsize=5)
    ax.bar(x + width/2, pre["Mean"], width, yerr=pre["Std"], label='Preprocessed', capsize=5)

    ax.set_title(f"{metric} Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel("Mean ± Std")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("metric_comparison.png", dpi=300)
'''
# Load SAM and SAM2 performance
sam = pd.read_csv("/vol/bitbucket/nc624/sam/segment-anything/outputs/csv files/PerformanceSummary_original.csv")
sam2 = pd.read_csv("/vol/bitbucket/nc624/sam2/outputs/CSV_Files/performance_nonProcessedSummary.csv")

# Set up plot
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
metrics = ["DICE", "Hausdorff95"]
phases = ["End Diastole", "End Systole", "Overall"]
x = np.arange(len(phases))
width = 0.35

# Use pastel colors
pastel_sam = "#a1c9f4"   # Light blue
pastel_sam2 = "#ffb482"  # Light orange

for i, metric in enumerate(metrics):
    ax = axs[i]
    data_sam = sam[sam["Metric"] == metric].reset_index(drop=True)
    data_sam2 = sam2[sam2["Metric"] == metric].reset_index(drop=True)

    # Zero out std for "Overall" (index 2)
    sam_std = data_sam["Std"].copy()
    sam_std[2] = 0

    sam2_std = data_sam2["Std"].copy()
    sam2_std[2] = 0

    # Plot bars
    ax.bar(x - width/2, data_sam["Mean"], width, 
           yerr=sam_std, label="SAM", capsize=5, 
           color=pastel_sam, edgecolor='black', linewidth=1.5)

    ax.bar(x + width/2, data_sam2["Mean"], width, 
           yerr=sam2_std, label="SAM2", capsize=5, 
           color=pastel_sam2, edgecolor='black', linewidth=1.5)

    ax.set_title(f"{metric} Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=12)
    ax.set_ylabel("Mean ± Std", fontsize=12)
    ax.legend()
    ax.grid(axis='y', linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

# Add a figure-wide title
plt.suptitle("Baseline Performance Comparison: SAM vs SAM2", fontsize=16, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("sam_vs_sam2_comparison.png", dpi=300)
# plt.show()

