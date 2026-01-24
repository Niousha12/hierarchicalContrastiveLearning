import os
import json

DATA_ROOT = "."
LIST_EVAL_PATH = os.path.join(DATA_ROOT, "list_eval_partition.txt")

# Where to save the output JSON files
OUT_DIR = "."  # current directory; change if you like


def get_category_from_path(image_rel_path):
    """
    Example image_rel_path:
      img/WOMEN/Dresses/id_00000002/02_1_front.jpg
    We want "Dresses".
    """
    parts = image_rel_path.split("/")
    # [ 'img', 'WOMEN', 'Dresses', 'id_00000002', '02_1_front.jpg' ]
    if len(parts) < 3:
        raise ValueError(f"Unexpected path format: {image_rel_path}")
    return parts[2]


def main():
    with open(LIST_EVAL_PATH, "r") as f:
        lines = f.readlines()

    # Skip the first two header lines (DeepFashion format)
    data_lines = [l.strip() for l in lines[2:] if l.strip()]

    train_imgs, train_cats = [], []
    val_imgs, val_cats = [], []  # gallery → val
    test_imgs, test_cats = [], []  # query  → test

    for line in data_lines:
        # image_name  item_id  eval_status
        # e.g.: img/WOMEN/Dresses/id_00000002/02_1_front.jpg 00000002 train
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Unexpected line format: {line}")
        img_rel, item_id, status = parts

        image_path = 'deepfashion/' + img_rel  # e.g. "img/WOMEN/Dresses/..."

        category = get_category_from_path(img_rel)

        if status == "train":
            train_imgs.append(image_path)
            train_cats.append(category)
        elif status == "gallery":
            val_imgs.append(image_path)  # gallery → val
            val_cats.append(category)
        elif status == "query":
            test_imgs.append(image_path)  # query → test
            test_cats.append(category)
        else:
            raise ValueError(f"Unknown status: {status}")

    os.makedirs(OUT_DIR, exist_ok=True)

    def save_listfile(filename, images, categories):
        assert len(images) == len(categories)
        obj = {"images": images, "categories": categories}
        out_path = os.path.join(OUT_DIR, filename)
        with open(out_path, "w") as f:
            json.dump(obj, f, indent=2)
        print(f"Saved {len(images)} entries to {out_path}")

    save_listfile("train_listfile.json", train_imgs, train_cats)
    save_listfile("val_listfile.json", val_imgs, val_cats)
    save_listfile("test_listfile.json", test_imgs, test_cats)


if __name__ == "__main__":
    main()
