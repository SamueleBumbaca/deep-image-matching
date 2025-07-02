# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# This script is based on an original implementation by True Price.

import sqlite3
import sys
import numpy as np
import argparse
import os
from PIL import Image, ExifTags
import csv

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(
    MAX_IMAGE_ID
)

CREATE_POSE_PRIORS_TABLE = """CREATE TABLE IF NOT EXISTS pose_priors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    position BLOB,
    coordinate_system INTEGER NOT NULL,
    position_covariance BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join(
    [
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_POSE_PRIORS_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_NAME_INDEX,
    ]
)


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tobytes()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


def get_exif_data(img):
    exif_data = {}
    try:
        info = img._getexif()
        if info is not None:
            for tag, value in info.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                exif_data[decoded] = value
    except AttributeError:
        pass
    return exif_data


def create_camera_model(
    images_folder_path, model=2, hard_coded_focal_length=None
):  # model=2 is the SIMPLE_RADIAL distortion model
    images_basename = os.listdir(images_folder_path)
    images_path = [os.path.join(images_folder_path, img) for img in images_basename]
    cameras = {"models": {}}

    for img_path, img_basename in zip(images_path, images_basename):
        img = Image.open(img_path)
        width, height = img.size

        # Read EXIF data
        exif_data = get_exif_data(img)
        camera_model = exif_data.get("Model", "")
        focal_length = exif_data.get("FocalLength", None)
        pixel_height = exif_data.get("PixelXDimension", None)
        pixel_width = exif_data.get("PixelYDimension", None)

        # Set intrinsic parameters based on camera model
        if (
            focal_length is not None
            and pixel_height is not None
            and pixel_width is not None
        ):
            fx = focal_length / pixel_width
            fy = focal_length / pixel_height
        elif (
            camera_model == "/base/soc/i2c0mux/i2c@1/imx219@10"
        ):  # Example of custom camera (Raspberry Pi Camera V2) without EXIF data
            focal_length = 2.8  # Focal Length: 2.8mm
            pixel_height = 0.00112  # Pixel size: 1.12 µm (H)
            pixel_width = 0.00112  # Pixel size: 1.12 µm (V)
            fx = focal_length / pixel_width
            fy = focal_length / pixel_height
        else:
            # Example intrinsic parameters (these should be calculated or provided)
            fx = 0.0
            fy = fx
        if hard_coded_focal_length is not None:
            fx = hard_coded_focal_length
            fy = fx
        cx = width / 2
        cy = height / 2
        if model == 2:  #'SIMPLE_RADIAL'
            distortion = [0.0]  # TODO: Calculate distortion parameters (OpenCV)
            params = (fx, fy, cx, cy, *distortion)
        else:
            NotImplementedError(
                "Only RADIAL distortion model is supported"
            )  # TODO: Implement other distortion models

        width_height = f"{width}_{height}"
        params_string = ",".join(map(str, params))

        if model not in cameras["models"]:
            cameras["models"][model] = {}
        if "width_height" not in cameras["models"][model]:
            cameras["models"][model] = {"width_height": {}}
        if width_height not in cameras["models"][model]["width_height"]:
            cameras["models"][model]["width_height"] = {
                width_height: {"value": (width, height)}
            }
        if "params" not in cameras["models"][model]["width_height"][width_height]:
            cameras["models"][model]["width_height"][width_height]["params"] = {}
        if (
            params_string
            not in cameras["models"][model]["width_height"][width_height]["params"]
        ):
            cameras["models"][model]["width_height"][width_height]["params"][
                params_string
            ] = {"value": params}
            cameras["models"][model]["width_height"][width_height]["params"][
                params_string
            ]["images"] = []

        cameras["models"][model]["width_height"][width_height]["params"][params_string][
            "images"
        ].append(img_basename)

    return cameras


class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = lambda: self.executescript(
            CREATE_DESCRIPTORS_TABLE
        )
        self.create_images_table = lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_pose_priors_table = lambda: self.executescript(
            CREATE_POSE_PRIORS_TABLE
        )
        self.create_two_view_geometries_table = lambda: self.executescript(
            CREATE_TWO_VIEW_GEOMETRIES_TABLE
        )
        self.create_keypoints_table = lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(
        self,
        model,
        width,
        height,
        params,
        prior_focal_length=False,
        camera_id=None,
    ):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (
                camera_id,
                model,
                width,
                height,
                array_to_blob(params),
                prior_focal_length,
            ),
        )
        return cursor.lastrowid

    def add_image(
        self,
        name,
        camera_id,
        image_id=None,
    ):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?)", (image_id, name, camera_id)
        )
        return cursor.lastrowid

    def add_pose_prior(
        self, image_id, position, coordinate_system=1, position_covariance=None
    ):
        position = np.asarray(position, dtype=np.float64)
        if position_covariance is None:
            position_covariance = np.full((3, 3), np.nan, dtype=np.float64)
        # Check if the image_id already exists
        cursor = self.execute(
            "SELECT COUNT(*) FROM pose_priors WHERE image_id = ?", (image_id,)
        )
        count = cursor.fetchone()[0]

        if count > 0:
            # Update the existing entry
            self.execute(
                "UPDATE pose_priors SET position = ?, coordinate_system = ?, position_covariance = ? WHERE image_id = ?",
                (
                    array_to_blob(position),
                    coordinate_system,
                    array_to_blob(position_covariance),
                    image_id,
                ),
            )
        else:
            # Insert a new entry
            self.execute(
                "INSERT INTO pose_priors VALUES (?, ?, ?, ?)",
                (
                    image_id,
                    array_to_blob(position),
                    coordinate_system,
                    array_to_blob(position_covariance),
                ),
            )

    def add_keypoints(self, image_id, keypoints):
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),),
        )

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),),
        )

    def add_matches(self, image_id1, image_id2, matches):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),),
        )

    def add_two_view_geometry(
        self,
        image_id1,
        image_id2,
        matches,
        F=np.eye(3),
        E=np.eye(3),
        H=np.eye(3),
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3),
        config=2,
    ):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,)
            + matches.shape
            + (
                array_to_blob(matches),
                config,
                array_to_blob(F),
                array_to_blob(E),
                array_to_blob(H),
                array_to_blob(qvec),
                array_to_blob(tvec),
            ),
        )


def main(
    database_path,
    images_folder_path,
    camera_coords_path,
    focal_length: float = None,
    update_camera_pose_priors: bool = False,
):
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    # For convenience, try creating all the tables upfront.
    db.create_tables()

    if not update_camera_pose_priors:
        # Create cameras from images.
        cameras = create_camera_model(
            images_folder_path, model=4, hard_coded_focal_length=focal_length
        )

        for model in cameras["models"].keys():
            for width_height in cameras["models"][model]["width_height"].keys():
                width, height = cameras["models"][model]["width_height"][width_height][
                    "value"
                ]
                for params_string in cameras["models"][model]["width_height"][
                    width_height
                ]["params"].keys():
                    params = cameras["models"][model]["width_height"][width_height][
                        "params"
                    ][params_string]["value"]
                    params = np.array(
                        params, dtype=np.float64
                    )  # Ensure params is a NumPy array
                    print(
                        f"Adding camera model: {model}, width: {width}, height: {height}, params: {params}"
                    )
                    camera_id = db.add_camera(model, width, height, params)
                    cameras["models"][model]["width_height"][width_height]["params"][
                        params_string
                    ]["camera_id"] = camera_id

        # Create images.
        images = {"id": {}}

        for model in cameras["models"].keys():
            for width_height in cameras["models"][model]["width_height"].keys():
                for params_string in cameras["models"][model]["width_height"][
                    width_height
                ]["params"].keys():
                    camera_id = cameras["models"][model]["width_height"][width_height][
                        "params"
                    ][params_string]["camera_id"]
                    for img_basename in cameras["models"][model]["width_height"][
                        width_height
                    ]["params"][params_string]["images"]:
                        image_id = db.add_image(img_basename, camera_id)
                        images["id"][
                            image_id
                        ] = img_basename  # Correctly update the images dictionary

    else:
        id_images_cam = db.execute("SELECT * FROM images").fetchall()
        images = {"id": {}}
        for id_image_cam in id_images_cam:
            id_image = id_image_cam[0]
            name = id_image_cam[1]
            images["id"][id_image] = name

    # Create pose_priors.
    with open(camera_coords_path, "r") as f:
        reader = csv.reader(f, delimiter=",")

        # Skip the first two lines
        next(reader)
        next(reader)

        for row in reader:
            img_basename = str(row[0])
            print(f"Processing image: {img_basename}")
            print(f"Row: {row}")
            if row[2] and row[3] and row[4]:  # Check if the values are not empty
                x, y, z = float(row[2]), float(row[3]), float(row[4])
                accuracy_xyz = float(row[8])
                position = np.array([x, y, z])
                position_covariance = np.diag(
                    [accuracy_xyz**2, accuracy_xyz**2, accuracy_xyz**2]
                )
                image_id_list = [
                    k for k, v in images["id"].items() if v == img_basename
                ]
                if image_id_list:
                    image_id = image_id_list[0]  # Extract the single integer value
                    print(
                        f"Adding pose prior for image: {img_basename}, position: {position}"
                    )
                    print(f"Image ID: {image_id}")
                    db.add_pose_prior(
                        image_id=image_id,
                        position=position,
                        position_covariance=position_covariance,
                    )
            else:
                print(f"Skipping row with missing values: {row}")

    db.commit()

    # Clean up.

    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="WriteCOLMAPDatabase",
        description="Create the project database.db by images and camera prior information",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""Example: python WriteCOLMAPDatabase.py -proj /home/user/project
                                                The images folder (/home/user/project/images) should be present in the project folder
                                                The camera_coords.csv file (/home/user/project/camera_coords.csv) should be present in the project folder
                                                camera_coords.csv should have the Agisoft Metashape format:
                                                # CoordinateSystem: LOCAL_CS["Local Coordinates (m)",LOCAL_DATUM["Local Datum",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]
                                                Label,Enable,X,Y,Z,Yaw,Pitch,Roll,Accuracy_X/Y/Z_(m),Accuracy_Yaw/Pitch/Roll_(deg),Error_(m),X_error,Y_error,Z_error,Error_(deg),Yaw_error,Pitch_error,Roll_error,X_est,Y_est,Z_est,Yaw_est,Pitch_est,Roll_est,X_var,Y_var,Z_var,Yaw_var,Pitch_var,Roll_var""",
    )
    parser.add_argument(
        "-proj",
        "--colmap_project_folder_path",
        type=str,
        required=True,
        help="Path to the COLMAP project folder, e.g. /home/user/project",
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        required=False,
        action="store_true",
        help="Overwrite the existing database.db file",
    )

    parser.add_argument(
        "-update_cam",
        "--update_camera_pose_priors",
        required=False,
        action="store_true",
        help="Update the existing database.db file",
    )

    parser.add_argument(
        "-f",
        "--focal_length",
        required=False,
        type=float,
        help="Hard-coded focal length of the camera",
    )

    args = parser.parse_args()
    project_path = args.colmap_project_folder_path
    overwrite = args.overwrite
    focal_length = args.focal_length
    update_camera_pose_priors = args.update_camera_pose_priors

    if overwrite and update_camera_pose_priors:
        RuntimeError(
            "Cannot overwrite and update the project at the same time. Choose one option between -o and -update_cam"
        )

    database_path = os.path.join(project_path, "database.db")
    if (
        os.path.exists(database_path)
        and not overwrite
        and not update_camera_pose_priors
    ):
        FileExistsError(
            "database path already exists -- will not modify it. Use -o to overwrite it"
        )
    elif os.path.exists(database_path) and overwrite and not update_camera_pose_priors:
        os.remove(database_path)
    elif os.path.exists(database_path) and update_camera_pose_priors and not overwrite:
        pass
    else:
        RuntimeError("database path not found")

    images_folder_path = os.path.join(project_path, "images")
    if not os.path.exists(images_folder_path):
        RuntimeError("images folder not found in the project folder")

    camera_coords_path = os.path.join(project_path, "camera_coords.csv")
    if not os.path.exists(camera_coords_path):
        FileExistsError("camera_coords.csv file not found in the project folder")

    imgs_basename = os.listdir(images_folder_path)
    if len(imgs_basename) == 0:
        RuntimeError("No images found in the images folder")

    main(
        database_path,
        images_folder_path,
        camera_coords_path,
        focal_length,
        update_camera_pose_priors,
    )
