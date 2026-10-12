import argparse

from utils import read_jsonl, write_jsonl


def verify(idx, triplet, detection):
    """Keep present objects that were detected and absent objects that were not; drop the image if either list empties."""
    present = [o for o in triplet["present"] if o in detection["present"]]
    no_exist = [o for o in triplet["no_exist"] if o not in detection["no_exist"]]
    if not present or not no_exist:
        return None
    return {**triplet, "present": present, "no_exist": no_exist, "idx": idx}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter images by Grounding-DINO detections.")
    parser.add_argument("--triplets", required=True, help="Output of 1_triplets.py.")
    parser.add_argument("--detections", required=True, help="Output of 2_2_grounding.py.")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    triplets = read_jsonl(args.triplets)
    detections = read_jsonl(args.detections)
    kept = [verify(i, t, d) for i, (t, d) in enumerate(zip(triplets, detections))]
    write_jsonl([item for item in kept if item is not None], args.output_path)
