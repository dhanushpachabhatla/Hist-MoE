from huggingface_hub import snapshot_download, login
import os
from dotenv import load_dotenv
load_dotenv()
# ---- LOGIN (safe) ----
login(token=os.getenv("HF_TOKEN"))

# ---- PATHS ----
DATA_DIR = "hest1k-dataset/hest_multicancer"
WSI_DIR = os.path.join(DATA_DIR, "wsis")

os.makedirs(WSI_DIR, exist_ok=True)

# ---- REQUIRED IDS ----
required_ids = ["TENX155","TENX156","TENX171","TENX175","TENX189","TENX190"]

# ---- CHECK EXISTING ----
existing_files = os.listdir(WSI_DIR)
existing_ids = [f.split(".")[0] for f in existing_files if f.endswith(".tif")]

missing_ids = [id for id in required_ids if id not in existing_ids]

print("Existing:", existing_ids)
print("Missing:", missing_ids)

# ---- DOWNLOAD ONLY MISSING ----
if len(missing_ids) > 0:

    print("Downloading missing WSIs...")

    patterns = []
    for id in missing_ids:
        patterns.append(f"**/{id}.tif")

    snapshot_download(
        repo_id="MahmoodLab/hest",
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=DATA_DIR
    )

    print("Download complete!")

else:
    print("All WSIs already present ✅")