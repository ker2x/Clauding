# How to Train on Google Colab

This guide explains how to run the training script on Google Colab's free GPUs.

## Step 1: Initialize Colab
1.  Go to [Google Colab](https://colab.research.google.com/).
2.  Click **New Notebook**.
3.  Go to **Runtime** > **Change runtime type**.
4.  Select **T4 GPU** (or better if you have Pro) and click **Save**.

## Step 2: Setup Workspace
Copy and run the following code in the first cell to clone your code and install dependencies.
*(Replace `<YOUR_REPO_URL>` with your actual GitHub repository URL).*

```python
# 1. Clone repository
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git project

# 2. Enter directory
%cd project/checkers

# 3. Install dependencies
!pip install -r requirements.txt
```

## Step 3: Start Training
Run the training script. We disable visualization since Colab is headless.

```python
# Run training for 100 iterations (adjust as needed)
!python scripts/train.py --iterations 100 --device cuda
```

## Step 4: Save Your Progress
Colab deletes files when the runtime disconnects. You **must** download your checkpoints or save them to Google Drive.

### Option A: Download Manually
Use the file browser on the left to navigate to `project/checkers/checkpoints` and download `best_model.pt`.

### Option B: Auto-Save to Google Drive (Recommended)
Mount Google Drive and symlink the checkpoint directory so everything is saved automatically.

**Run this BEFORE training:**
```python
from google.colab import drive
drive.mount('/content/drive')

# Create a folder in your Drive
!mkdir -p /content/drive/MyDrive/CheckersAI/checkpoints

# Link project checkpoint dir to Drive
!rm -rf checkpoints
!ln -s /content/drive/MyDrive/CheckersAI/checkpoints checkpoints

print("Checkpoints will now save directly to Google Drive!")
```
