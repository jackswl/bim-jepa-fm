import os
import argparse
import trimesh
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# --- Folder lists for the two BIMCompNet usage modes ---
PRETRAIN_BASE_NAMES = [
    "Malls", "Museum", "Hospital_A", "Hospital_B", "Hospital_C",
    "EquipmentRoom", "Factory", "Hotels", "Layout", "Metro",
    "Office", "Public", "Residential", "School", "Theater", "Villa",
]

CLASSIFICATION_SAMPLE_SIZES = [100, 500, 1000, 5000]
CLASSIFICATION_SPLITS = ["train", "test"]


def convert_mesh_to_pointcloud(mesh_path, num_points):
    """
    Loads a mesh, converts to point cloud, normalizes (center + unit sphere).
    """
    try:
        # force='mesh' prevents crashes on complex BIM objects
        mesh = trimesh.load(mesh_path, force='mesh', process=False)

        points, _ = trimesh.sample.sample_surface(mesh, num_points)

        points = points - points.mean(axis=0)
        max_dist = np.max(np.linalg.norm(points, axis=1))

        if max_dist > 0:
            points = points / max_dist

        return points.astype(np.float32)

    except Exception as e:
        print(f"Error processing {mesh_path}: {e}")
        return None


def process_single_file(args):
    """
    Processes one file. SKIPS if output .npy already exists and is non-empty.
    """
    mesh_path, input_root, output_root, num_points = args

    relative_path = os.path.relpath(mesh_path, input_root)
    output_filename = os.path.splitext(relative_path)[0] + '.npy'
    output_path = os.path.join(output_root, output_filename)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return "SKIPPED"

    points = convert_mesh_to_pointcloud(mesh_path, num_points)

    if points is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, points)
        return "PROCESSED"

    return "FAILED"


def process_multiple_datasets(dataset_list, num_points, num_workers):
    print(f"Firing up {num_workers} parallel workers...")

    for input_dir, output_dir in dataset_list:
        print(f"\n--- Processing: {os.path.basename(input_dir)} ---")

        if not os.path.isdir(input_dir):
            print(f"Skipping missing directory: {input_dir}")
            continue

        tasks = []
        print("Scanning files...")
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith((".obj", ".ply")):
                    full_path = os.path.join(root, file)
                    tasks.append((full_path, input_dir, output_dir, num_points))

        print(f"Found {len(tasks)} candidates.")

        processed_count = 0
        skipped_count = 0
        failed_count = 0

        with Pool(processes=num_workers) as pool:
            for result in tqdm(pool.imap_unordered(process_single_file, tasks),
                               total=len(tasks), desc="Progress"):
                if result == "PROCESSED":
                    processed_count += 1
                elif result == "SKIPPED":
                    skipped_count += 1
                elif result == "FAILED":
                    failed_count += 1

        print(f"Done. {processed_count} new files created. "
              f"{skipped_count} skipped. {failed_count} failed.")


def build_pretrain_datasets(base_path, num_points):
    return [
        (
            os.path.join(base_path, f"{name}_new"),
            os.path.join(base_path, f"{name}_pointclouds_{num_points}"),
        )
        for name in PRETRAIN_BASE_NAMES
    ]


def build_classification_datasets(base_path, num_points):
    # num_points is unused in the folder name to match the upstream layout
    # (BIMCompNet_<N>_<split>_npy). If you run with multiple num_points, rename
    # or pre-clear the output folder between runs.
    del num_points
    return [
        (
            os.path.join(base_path, f"BIMCompNet_{n}_{split}"),
            os.path.join(base_path, f"BIMCompNet_{n}_{split}_npy"),
        )
        for n in CLASSIFICATION_SAMPLE_SIZES
        for split in CLASSIFICATION_SPLITS
    ]


def resolve_workers(requested):
    if requested is not None:
        return requested
    # On clusters, os.cpu_count() may return the node total even if you only
    # requested a subset; prefer PBS_NCPUS when present.
    try:
        return int(os.environ["PBS_NCPUS"])
    except (KeyError, ValueError):
        return cpu_count()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert BIMCompNet .obj meshes into normalised .npy point clouds "
                    "in parallel."
    )
    parser.add_argument(
        "--base-path",
        default=os.environ.get("BIMCOMP_DATA_DIR", "./bimcompnet"),
        help="Base directory holding the input folders and where output folders "
             "will be created (env: BIMCOMP_DATA_DIR; default: ./bimcompnet).",
    )
    parser.add_argument(
        "--mode",
        choices=["pretrain", "classification"],
        default="pretrain",
        help="'pretrain' processes the 16 building folders (Museum_new, Malls_new, ...). "
             "'classification' processes the BIMCompNet 100/500/1000/5000 train/test splits.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=4096,
        help="Number of surface points to sample per mesh (default: 4096).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: $PBS_NCPUS or cpu_count()).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    num_workers = resolve_workers(args.workers)

    if args.mode == "pretrain":
        datasets = build_pretrain_datasets(args.base_path, args.num_points)
    else:
        datasets = build_classification_datasets(args.base_path, args.num_points)

    process_multiple_datasets(datasets, args.num_points, num_workers)
    print("\nAll datasets processed.")


if __name__ == "__main__":
    main()
