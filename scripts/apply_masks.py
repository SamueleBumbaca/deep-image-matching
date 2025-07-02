import os
from pathlib import Path
from PIL import Image
import numpy as np
import argparse


def main(
    input_dir: Path,
    output_dir: Path,
    img_folder: Path,
    mask_folder: Path,
    copy_only_images_with_mask: bool,
):
    # check if the output directory exists
    if os.path.exists(input_dir):
        input_dir = Path(input_dir)
        img_path = input_dir / img_folder
        images = os.listdir(img_path)
    else:
        raise ValueError("Input directory does not exist")
    # check if all the images have the same dimension, if not raise an error
    sizes = []
    for img_name in images:
        img = Image.open(img_path / img_name)
        sizes.append(img.size)
    if len(set(sizes)) != 1:
        raise ValueError("Images have different dimensions")
    size = sizes[0]
    # do the same for the masks
    if os.path.exists(input_dir):
        mask_path = input_dir / mask_folder
        masks = os.listdir(mask_path)
    else:
        raise ValueError("Input directory does not exist")
    # check if all the masks have the same dimension, if not raise an error
    mask_sizes = []
    for mask_name in masks:
        mask = Image.open(mask_path / mask_name)
        mask_sizes.append(mask.size)
    if len(set(mask_sizes)) != 1:
        raise ValueError("Masks have different dimensions")
    mask_size = mask_sizes[0]
    if mask_size != size:
        raise ValueError("Images and masks have different dimensions")
    # pair the images and masks in a dictionary, they must paired by the basename
    img_mask = {}
    for img_name in images:
        img_mask[img_name] = img_name  # .replace(".jpg", ".png")
    # check if all the images have a mask
    if copy_only_images_with_mask == None and set(images) != set(masks):
        raise ValueError("Some images do not have a mask")
    # check if the output directory exists
    out_dir = Path(output_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # write the images and masks in the output directory using the dictionary
    for img_name, mask_name in img_mask.items():
        # if the images do not have a mask and the flag is set to True, skip the image
        if copy_only_images_with_mask and mask_name not in masks:
            continue
        # if the images do not have a mask and the flag is set to False, copy the image
        if not copy_only_images_with_mask and mask_name not in masks:
            img = Image.open(img_path / img_name)
            img.save(out_dir / img_name)
            continue
        # if the images have a mask, mask the image and save it
        img = Image.open(img_path / img_name)
        mask = Image.open(mask_path / mask_name)
        img = np.array(img)
        mask = np.array(mask) > 0  # binarize the mask
        img[mask == 0] = 0
        img = Image.fromarray(img)
        img.save(out_dir / img_name)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_folder", type=str, required=True)
    parser.add_argument("--mask_folder", type=str, required=True)
    parser.add_argument("--copy_only_images_with_mask", type=bool, default=None)
    args = parser.parse_args()
    main(
        args.input_dir,
        args.output_dir,
        args.img_folder,
        args.mask_folder,
        args.copy_only_images_with_mask,
    )
