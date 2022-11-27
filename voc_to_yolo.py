import glob
import os
import random
import shutil
import xml.etree.ElementTree as ET

from tqdm import tqdm

INPUT_IMAGE_DIR = "/data/datasets/tdt17/RDD2022/ALL/train/images"
INPUT_ANNOTATIONS_DIR = "/data/datasets/tdt17/RDD2022/ALL/train/annotations/xmls"
OUTPUT_DIR = "/lhome/mathiawo/work/RDD2022_YOLO"
ALL_LABLES_DIR = f"{OUTPUT_DIR}/ALL/labels"

POTHOLE_DATASET_DIR = "/data/datasets/pothole_dataset"

FOLDER_SPLITS = {"train": 1, "val": 0.0, "test": 0.0}

CLASSES = ["D00", "D10", "D20", "D40"]


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def convert():
    print("Converting...")
    skipped_classes = set()

    # create the labels folder (output directory)
    os.makedirs(ALL_LABLES_DIR, exist_ok=True)

    # identify all the xml files in the annotations folder (input directory)
    files = glob.glob(os.path.join(INPUT_ANNOTATIONS_DIR, "*.xml"))
    # loop through each
    for file in tqdm(files):
        basename = os.path.basename(file)
        filename = os.path.splitext(basename)[0]
        # check if the label contains the corresponding image file
        if not os.path.exists(os.path.join(INPUT_IMAGE_DIR, f"{filename}.jpg")):
            print(f"{filename} image does not exist!")
            continue

        result = []

        # parse the content of the xml file
        tree = ET.parse(file)
        root = tree.getroot()
        width = int(float(root.find("size").find("width").text))
        height = int(float(root.find("size").find("height").text))

        for obj in root.findall("object"):
            label = obj.find("name").text
            # check for new classes and append to list
            if label not in CLASSES:
                skipped_classes.add(label)
                continue

            index = CLASSES.index(label)
            pil_bbox = [int(float(x.text)) for x in obj.find("bndbox")]
            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
            # convert data to string
            bbox_string = " ".join([str(x) for x in yolo_bbox])
            result.append(f"{index} {bbox_string}")

        if result:
            # generate a YOLO format text file for each xml file
            with open(
                os.path.join(ALL_LABLES_DIR, f"{filename}.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("\n".join(result))

    print("Done!")
    print(f"Used classes: {CLASSES}")
    print(f"Skipped classes: {skipped_classes}")


def copyfiles(image_file_path, label_dir, image_dir):
    basename = os.path.basename(image_file_path)
    filename = os.path.splitext(basename)[0]

    # copy image
    dest = os.path.join(image_dir, f"{filename}.jpg")
    shutil.copyfile(image_file_path, dest)

    # copy annotations, if they exist
    src = os.path.join(ALL_LABLES_DIR, f"{filename}.txt")
    if os.path.exists(src):
        dest = os.path.join(label_dir, f"{filename}.txt")
        shutil.copyfile(src, dest)


def split_datasets():
    lower_limit = 0
    image_file_paths = glob.glob(os.path.join(INPUT_IMAGE_DIR, "*.jpg"))

    random.shuffle(image_file_paths)

    assert (
        sum([FOLDER_SPLITS[x] for x in FOLDER_SPLITS]) == 1.0
    ), "Split proportion is not equal to 1.0"

    for folder in FOLDER_SPLITS:
        print(f"Creating {folder} dataset")
        label_dir = f"{OUTPUT_DIR}/labels/{folder}"
        image_dir = f"{OUTPUT_DIR}/images/{folder}"
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        limit = round(len(image_file_paths) * FOLDER_SPLITS[folder])
        for image_file_path in tqdm(
            image_file_paths[lower_limit : lower_limit + limit]
        ):
            copyfiles(image_file_path, label_dir, image_dir)
        lower_limit = lower_limit + limit


def add_pothole_dataset():
    for split in ["train", "valid", "test"]:
        print(f"Adding pothole dataset {split} split")
        input_label_dir = f"{POTHOLE_DATASET_DIR}/labels/{split}"
        input_image_dir = f"{POTHOLE_DATASET_DIR}/images/{split}"

        temp_label_dir = f"temp/labels/{split}"
        os.makedirs(temp_label_dir, exist_ok=True)

        # copy all labels from input_label_dir to temp_label_dir
        for label_file in glob.glob(os.path.join(input_label_dir, "*.txt")):
            shutil.copyfile(
                label_file, os.path.join(temp_label_dir, os.path.basename(label_file))
            )

        # for all files in temp_label_dir, replace the first character in every line with 3
        for label_file in glob.glob(os.path.join(temp_label_dir, "*.txt")):
            with open(label_file, "r") as f:
                lines = f.readlines()
            with open(label_file, "w") as f:
                for line in lines:
                    f.write(f"3{line[1:]}")

        target_folder = "val" if split == "valid" else split
        output_label_dir = f"{OUTPUT_DIR}/labels/{target_folder}"
        output_image_dir = f"{OUTPUT_DIR}/images/{target_folder}"

        # copy all labels from temp_label_dir to output_label_dir
        for label_file in glob.glob(os.path.join(temp_label_dir, "*.txt")):
            shutil.copyfile(
                label_file, os.path.join(output_label_dir, os.path.basename(label_file))
            )

        # copy all images from input_image_dir to output_image_dir, both jpg and JPG
        for image_file in tqdm(glob.glob(os.path.join(input_image_dir, "*.[jJ][pP][gG]"))):
            shutil.copyfile(
                image_file, os.path.join(output_image_dir, os.path.basename(image_file))
            )

        # remove temp_label_dir
        shutil.rmtree(temp_label_dir)


if __name__ == "__main__":
    print("Clearing output directory...")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    convert()
    split_datasets()
    # add_pothole_dataset()
