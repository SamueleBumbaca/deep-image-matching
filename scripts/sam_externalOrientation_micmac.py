import csv
import xml.etree.ElementTree as ET


def create_initial_orientations_xml(input_csv, output_xml):
    root = ET.Element("Global")
    chantier = ET.SubElement(root, "ChantierDescripteur")
    loc_cam_db = ET.SubElement(chantier, "LocCamDataBase")
    keyed_names = ET.SubElement(chantier, "KeyedNamesAssociations")

    with open(input_csv, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)  # Skip the first row with the coordinate system info
        headers = next(reader)  # Read the actual header row

        # Create a DictReader with the correct headers
        reader = csv.DictReader(csvfile, fieldnames=headers, delimiter=",")
        for row in reader:
            if row["Enable"] == "1":
                image_name = row["Label"]
                tx = row["X"]
                ty = row["Y"]
                tz = row["Z"]
                yaw = row["Yaw"]
                pitch = row["Pitch"]
                roll = row["Roll"]

                camera_entry = ET.SubElement(loc_cam_db, "CameraEntry")
                ET.SubElement(camera_entry, "Name").text = image_name
                ET.SubElement(camera_entry, "Pos").text = f"{tx} {ty} {tz}"
                ET.SubElement(camera_entry, "Ori").text = f"{yaw} {pitch} {roll}"

    tree = ET.ElementTree(root)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)


# Convert the initial orientations
create_initial_orientations_xml(
    "/mnt/d/CV/GEP/SAG24_C_6027_F3632_corn/2024_05_16/temp/T_3_B_1_camera_coords.csv",
    "/home/samuelebumbaca/repositories/deep-image-matching-sam/assets/example_plant/micmac_/MicMac-InitialOrientations.xml",
)
