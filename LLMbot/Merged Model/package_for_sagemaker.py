"""
Run this script from inside your MergedModel folder,
or update MERGED_MODEL_DIR to point to it.
"""
import os
import shutil
import tarfile

# Update this path to your MergedModel folder
MERGED_MODEL_DIR = "./MergedModel"  # or full path like r"C:\Users\...\MergedModel"

# Output tarball name
OUTPUT_FILE = "model.tar.gz"

# Path to your inference.py (put it next to this script)
INFERENCE_SCRIPT = "inference.py"


def create_sagemaker_package():
    # Verify paths exist
    if not os.path.exists(MERGED_MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {MERGED_MODEL_DIR}")

    if not os.path.exists(INFERENCE_SCRIPT):
        raise FileNotFoundError(f"inference.py not found. Put it next to this script.")

    # Create tarball
    print(f"Creating {OUTPUT_FILE}...")

    with tarfile.open(OUTPUT_FILE, "w:gz") as tar:
        # Add all model files
        for filename in os.listdir(MERGED_MODEL_DIR):
            filepath = os.path.join(MERGED_MODEL_DIR, filename)
            if os.path.isfile(filepath):
                print(f"  Adding: {filename}")
                tar.add(filepath, arcname=filename)

        # Add inference script to "code" subdirectory (SageMaker convention)
        print(f"  Adding: code/inference.py")
        tar.add(INFERENCE_SCRIPT, arcname="code/inference.py")

    # Get file size
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\nDone! Created {OUTPUT_FILE} ({size_mb:.1f} MB)")
    print(f"\nNext steps:")
    print(f"  1. Upload to S3: aws s3 cp {OUTPUT_FILE} s3://your-bucket/models/")
    print(f"  2. Deploy to SageMaker (see deployment script)")


if __name__ == "__main__":
    create_sagemaker_package()