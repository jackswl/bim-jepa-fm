import os
import argparse
import shutil
from pathlib import Path


# Default folder lists for the BIMCompNet release. Override on the CLI if your
# layout differs (comma-separated, no spaces).
DEFAULT_TEST_FOLDERS = [
    "BIMCompNet_100_test",
    "BIMCompNet_500_test",
    "BIMCompNet_1000_test",
    "BIMCompNet_5000_test",
]

DEFAULT_PRETRAIN_FOLDERS = [
    "Museum_new", "Malls_new", "Hospital_A_new", "Hospital_B_new", "Hospital_C_new",
    "EquipmentRoom_new", "Factory_new", "Hotels_new", "Layout_new", "Metro_new",
    "Office_new", "Public_new", "Residential_new", "School_new", "Theater_new", "Villa_new",
]


def get_all_filenames(base_path, subfolders):
    """
    Scan the given folders and return a SET of pretrain-style .obj filenames
    derived from the (possibly classification-style) test files. The classification
    naming has an extra trailing class token (e.g. 7_IfcBeam_0_IfcStructuralMember.obj)
    that we strip back to the pretrain form (7_IfcBeam_0.obj) so we can match it
    against the pretrain folders.
    """
    filenames = set()
    print("Step 1: Building Blacklist from Test Sets...")

    for folder in subfolders:
        path = Path(base_path) / folder
        if not path.exists():
            print(f"   Warning: Test folder not found: {folder}")
            continue

        print(f"   Scanning {folder}...")
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.obj'):
                    stem = os.path.splitext(f)[0]
                    parts = stem.split('_')
                    # Classification stem: [Project, SpecificType, Instance, BroadClass]
                    # Reduce to pretrain:  [Project, SpecificType, Instance]
                    if len(parts) >= 4 and parts[-1].startswith('Ifc'):
                        filenames.add('_'.join(parts[:-1]) + '.obj')
                    else:
                        filenames.add(f)

    print(f"   >>> Found {len(filenames)} unique files to banish.\n")
    return filenames


def quarantine_files(base_dir, quarantine_dir, test_folders, pretrain_folders):
    root = Path(base_dir)
    quarantine_root = Path(quarantine_dir)

    blacklist = get_all_filenames(base_dir, test_folders)

    if len(blacklist) == 0:
        print("No test files found! Have you downloaded the test sets yet?")
        return

    print("Step 2: Cleaning Pre-training Data...")

    total_moved = 0

    for folder in pretrain_folders:
        source_dir = root / folder

        if not source_dir.exists():
            continue

        # e.g. /_QUARANTINE/Museum_new/
        dest_dir = quarantine_root / folder
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"   Cleaning {folder}...")

        files_to_move = [f for f in os.listdir(source_dir) if f in blacklist]

        for fname in files_to_move:
            src_file = source_dir / fname
            dst_file = dest_dir / fname

            try:
                shutil.move(str(src_file), str(dst_file))
                total_moved += 1
            except Exception as e:
                print(f"Failed to move {fname}: {e}")

    print("-" * 40)
    print(f"DONE. Total files quarantined: {total_moved}")
    print(f"   Files are moved to: {quarantine_dir}")
    print("-" * 40)


def parse_csv_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move any pretrain .obj files that overlap with the classification "
                    "test sets into a quarantine directory, so they don't leak into pretraining."
    )
    parser.add_argument(
        "--base-dir",
        default=os.environ.get("BIMCOMP_DATA_DIR", "./bimcompnet"),
        help="Base directory containing both pretrain and test folders "
             "(env: BIMCOMP_DATA_DIR; default: ./bimcompnet).",
    )
    parser.add_argument(
        "--quarantine-dir",
        default=os.environ.get("BIMCOMP_QUARANTINE_DIR"),
        help="Directory to move overlapping files into "
             "(env: BIMCOMP_QUARANTINE_DIR; default: <base-dir>/_QUARANTINE).",
    )
    parser.add_argument(
        "--test-folders",
        type=parse_csv_list,
        default=DEFAULT_TEST_FOLDERS,
        help="Comma-separated list of test folder names. Defaults to the standard "
             "BIMCompNet 100/500/1000/5000 test splits.",
    )
    parser.add_argument(
        "--pretrain-folders",
        type=parse_csv_list,
        default=DEFAULT_PRETRAIN_FOLDERS,
        help="Comma-separated list of pretrain folder names to scan for overlap.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    quarantine_dir = args.quarantine_dir or os.path.join(args.base_dir, "_QUARANTINE")
    quarantine_files(
        base_dir=args.base_dir,
        quarantine_dir=quarantine_dir,
        test_folders=args.test_folders,
        pretrain_folders=args.pretrain_folders,
    )


if __name__ == "__main__":
    main()
