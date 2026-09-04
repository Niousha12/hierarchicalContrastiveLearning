"""
Generates the ImageNet supercategory hierarchy JSON required by train_imagenet.py.

The output is a JSON file mapping each ImageNet synset folder name (e.g. "n01440764")
to a supercategory string (its immediate WordNet hypernym).  This matches the
hierarchy used in the paper via the Robustness library (MadryLab).

Requirements:
    pip install nltk
    python -c "import nltk; nltk.download('wordnet')"

Usage:
    python scripts/prepare_imagenet_hierarchy.py \
        --imagenet-root /path/to/imagenet \
        --output data_processing/imagenet_hierarchy.json

The script scans the train/ subdirectory to discover all synset IDs, then walks
the WordNet hypernym tree to find a single-level supercategory for each class.
"""

import argparse
import json
import os


def get_wordnet():
    try:
        from nltk.corpus import wordnet as wn
        # Trigger a small lookup to verify the corpus is installed
        list(wn.synsets('dog'))
        return wn
    except LookupError:
        import nltk
        print("Downloading WordNet corpus...")
        nltk.download('wordnet')
        from nltk.corpus import wordnet as wn
        return wn


def synset_from_offset(wn, offset_str):
    """Convert an ImageNet synset ID like 'n01440764' to a WordNet Synset."""
    pos = offset_str[0]       # 'n' for noun
    offset = int(offset_str[1:])
    try:
        return wn.synset_from_pos_and_offset(pos, offset)
    except Exception:
        return None


def find_supercategory(synset, depth=1):
    """Walk up the hypernym tree by `depth` steps and return the name."""
    current = synset
    for _ in range(depth):
        hypernyms = current.hypernyms()
        if not hypernyms:
            break
        current = hypernyms[0]
    # Return the lemma name with underscores replaced by spaces
    return current.lemma_names()[0].replace('_', ' ')


def main():
    parser = argparse.ArgumentParser(description='Build ImageNet hierarchy JSON')
    parser.add_argument('--imagenet-root', required=True,
                        help='Root of ImageNet dataset (must contain a train/ subdir)')
    parser.add_argument('--output', default='data_processing/imagenet_hierarchy.json',
                        help='Output JSON path (default: data_processing/imagenet_hierarchy.json)')
    parser.add_argument('--depth', type=int, default=1,
                        help='How many hypernym levels to ascend for supercategory (default: 1)')
    args = parser.parse_args()

    train_dir = os.path.join(args.imagenet_root, 'train')
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Could not find train/ under {args.imagenet_root}. "
            "Make sure --imagenet-root points to the ILSVRC-2012 root directory."
        )

    synset_ids = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )
    print(f"Found {len(synset_ids)} classes in {train_dir}")

    wn = get_wordnet()

    hierarchy = {}
    missing = []
    for sid in synset_ids:
        synset = synset_from_offset(wn, sid)
        if synset is None:
            missing.append(sid)
            hierarchy[sid] = 'unknown'
            continue
        supercategory = find_supercategory(synset, depth=args.depth)
        hierarchy[sid] = supercategory

    if missing:
        print(f"Warning: could not resolve {len(missing)} synsets in WordNet: {missing[:10]}...")

    # Count unique supercategories
    unique_super = set(hierarchy.values())
    print(f"Mapped {len(synset_ids)} classes to {len(unique_super)} supercategories "
          f"at hypernym depth {args.depth}.")
    print("Sample mappings:")
    for sid in list(synset_ids)[:5]:
        print(f"  {sid} -> {hierarchy[sid]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(hierarchy, f, indent=2)
    print(f"\nHierarchy saved to: {args.output}")
    print("Pass this file to train_imagenet.py via --hierarchy-file")


if __name__ == '__main__':
    main()
