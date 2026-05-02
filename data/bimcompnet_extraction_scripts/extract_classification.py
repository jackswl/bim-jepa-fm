# Used for the BIMCompNet classification splits (100/500/1000/5000 train/test).
# Differs from extract_pretrain.py because the sampling dataset merged some classes
# into a single broad class. Naming convention: Instance_Class.obj (instead of just Instance.obj).


import os
import argparse
import requests
import zlib
import struct
import time
from remotezip import RemoteZip
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# Set from CLI args in main(); used inside the worker function.
MAX_RETRIES = 3


def download_raw_chunk(args):
    url, offset, comp_size, comp_type, dest_path = args

    last_error = "unknown"
    for _ in range(MAX_RETRIES):
        try:
            fetch_size = comp_size + 1024
            headers = {'Range': f'bytes={offset}-{offset + fetch_size}'}
            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                if r.status_code != 206:
                    continue
                data = r.content

            name_len = struct.unpack('<H', data[26:28])[0]
            extra_len = struct.unpack('<H', data[28:30])[0]
            header_size = 30 + name_len + extra_len
            compressed_data = data[header_size : header_size + comp_size]

            if comp_type == 0:
                final_data = compressed_data
            elif comp_type == 8:
                final_data = zlib.decompress(compressed_data, -15)
            else:
                return (False, "Unsupported Compression")

            with open(dest_path, 'wb') as f:
                f.write(final_data)
            return (True, None)

        except Exception as e:
            time.sleep(1)
            last_error = str(e)

    return (False, last_error)


def process_dataset(zip_url, folder_name, output_dir, max_workers):
    dest_path = Path(output_dir) / folder_name
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[{folder_name}] 1. Reading Index...")
    tasks = []
    skipped_count = 0

    try:
        with RemoteZip(zip_url) as zip_file:
            info_list = zip_file.infolist()
            print(f"[{folder_name}] Index downloaded. Scanning {len(info_list)} files...")

            for info in info_list:
                # We only want .obj files inside an /OBJ/ folder
                if info.filename.endswith('.obj') and '/OBJ/' in info.filename:
                    path_parts = info.filename.split('/')
                    try:
                        obj_index = path_parts.index('OBJ')

                        # Instance Name is immediately before OBJ (e.g., 0_IfcFan_1)
                        instance_name = path_parts[obj_index - 1]

                        # Class Name is immediately before Instance Name (e.g., IfcFlowMovingDevice)
                        if obj_index >= 2:
                            class_name = path_parts[obj_index - 2]
                            new_filename = f"{instance_name}_{class_name}.obj"
                        else:
                            new_filename = f"{instance_name}.obj"

                        save_location = dest_path / new_filename

                        if save_location.exists() and save_location.stat().st_size > 0:
                            skipped_count += 1
                            continue

                        tasks.append((
                            zip_url,
                            info.header_offset,
                            info.compress_size,
                            info.compress_type,
                            save_location
                        ))
                    except ValueError:
                        continue

        print(f"[{folder_name}] 2. Status: {skipped_count} Skipped (Exists), {len(tasks)} To Download.")

        if len(tasks) == 0:
            print(f"[{folder_name}] All files exist. Moving to next.")
            return

        print(f"[{folder_name}] 3. Extracting missing files with {max_workers} workers...")

        failed_files = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(download_raw_chunk, t): t[4].name for t in tasks}

            for future in tqdm(as_completed(future_to_file), total=len(tasks), unit="file", desc=folder_name):
                filename = future_to_file[future]
                success, error_msg = future.result()
                if not success:
                    failed_files.append((filename, error_msg))

        print(f"[{folder_name}] DONE.")

        if failed_files:
            print(f"\n[{folder_name}] FAILURE REPORT: The following {len(failed_files)} files could not be downloaded:")
            for fname, err in failed_files:
                print(f" - {fname} (Error: {err})")
            print(" These files may be corrupt in the source ZIP or persistently blocked.\n")
        else:
            print(f"[{folder_name}] All missing files successfully downloaded.\n")

    except Exception as e:
        print(f"[{folder_name}] CRITICAL ERROR: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream-download .obj files from a remote BIMCompNet classification zip "
                    "(named as Instance_Class.obj)."
    )
    parser.add_argument("url", help="Remote ZIP URL (e.g. https://host/BIMCompNet_100_train.zip).")
    parser.add_argument("folder", help="Destination folder name, created under --output-dir.")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("BIMCOMP_DATA_DIR", "./bimcompnet"),
        help="Base output directory (env: BIMCOMP_DATA_DIR; default: ./bimcompnet).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("NCPUS", 32)),
        help="Number of parallel download workers (env: NCPUS; default: 32).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of times to retry a failed chunk download (default: 3).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global MAX_RETRIES
    MAX_RETRIES = args.max_retries

    process_dataset(args.url, args.folder, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
