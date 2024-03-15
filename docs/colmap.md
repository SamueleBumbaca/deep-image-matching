# Use with COLMAP and pyCOLMAP

## COLMAP 

Deep-Image-Matching saves the results in a sqlite database, that database can be directly used with COLMAP to run the reconstruction.

To import the solution into COLMAP, open the COLAMP GUI and create a new project by "File" -> "New Project".
Then select the sqlite database file that was created by Deep-Image-Matching and the directory containing the images.

![COLMAP import](./assets/colamp1.png)

You can then use the `Database Management` tab to explore the matches for each image pair.

![COLMAP matches](./assets/colamp2.png)

## pyCOLMAP

pyCOLMAP is a python wrapper for COLMAP, it can be used to load and manipulate an existing COLMAP reconstruction.

To run the reconstruction with pyCOLMAP, you must install it first:
```bash
pip install pycolmap==0.6.1
```

and run Deep-Image-Matching without the `--skip-reconstruction` flag, to save the reconstruction is carried out by pycolmap directly.

```bash
python main.py --dir assets/example_cyprus --pipeline superpoint+lightglue
```

Then you can use the following code to load the reconstruction and manipulate it:

```python
import pycolmap
reconstruction = pycolmap.Reconstruction("path/to/reconstruction/dir")
print(reconstruction.summary())

for image_id, image in reconstruction.images.items():
    print(image_id, image)

for point3D_id, point3D in reconstruction.points3D.items():
    print(point3D_id, point3D)

for camera_id, camera in reconstruction.cameras.items():
    print(camera_id, camera)

reconstruction.write("path/to/reconstruction/dir/")
```