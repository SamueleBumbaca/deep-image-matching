from pathlib import Path
import numpy as np
import sqlite3
from deep_image_matching.utils.database import COLMAPDatabase
import pycolmap


def array_to_blob(array):
    """Convert a NumPy array to a BLOB."""
    return array.tobytes()


def image_pair_to_pair_id(image_id1, image_id2):
    """Compute the unique pair ID for a pair of image IDs."""
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * 2147483647 + image_id2


# Example matches data (replace with your actual matches data)
matches = np.array([[0, 1], [2, 3], [4, 5]])

# Path to the COLMAP database
database_path = Path("scripts/sam_custom/database_test/database.db")

# Open the database
db = COLMAPDatabase.connect(database_path)

# Add matches to the database
image_id1 = 1  # Replace with actual image ID
image_id2 = 2  # Replace with actual image ID

try:
    db.add_matches(image_id1, image_id2, matches)
except sqlite3.IntegrityError:
    # Handle the exception and update the existing matches
    print(
        f"Matches for image pair ({image_id1}, {image_id2}) already exist. Updating matches."
    )
    db.execute(
        "UPDATE matches SET data = ? WHERE pair_id = ?",
        (array_to_blob(matches), image_pair_to_pair_id(image_id1, image_id2)),
    )

# Commit the changes
db.commit()

# Close the database
db.close()

import pycolmap
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

db_path = Path("results_roma_bruteforce_quality_lowest/database.db")
poses_csv = "camera_coords.csv"

# Open the COLMAP database
db = pycolmap.Database()
db.open(db_path)
# Read cameras
cameras = {camera.camera_id: camera for camera in db.read_all_cameras()}
# Read images
images = db.read_all_images()
# Read keypoints
keypoints = {img.image_id: db.read_keypoints(img.image_id) for img in images}
# Read camera poses from CSV, skipping the first row
poses_df = pd.read_csv(poses_csv, skiprows=1)

keypoints = {img.image_id: db.read_keypoints(img.image_id) for img in images}
matches = {
    img_1.image_id: {
        img_2.image_id: db.read_matches(img_1.image_id, img_2.image_id)
        for img_2 in images
    }
    for img_1 in images
}

print("matches:")
print(matches)
