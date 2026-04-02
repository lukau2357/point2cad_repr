"""
Collect per-model results and upload to Google Drive via PyDrive2.

First-time setup:
  1. pip install pydrive2
  2. Create OAuth credentials at https://console.cloud.google.com/apis/credentials
     - Create OAuth 2.0 Client ID (Desktop app)
     - Download JSON → save as client_secrets.json in this directory
  3. Run the script — it will open a browser for authentication

Credentials are cached in `credentials.json`. Delete both `credentials.json`
and `client_secrets.json` when done to prevent security issues.
"""

import argparse
import glob as _glob
import os
import shutil
import tempfile

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR     = os.path.dirname(os.path.abspath(__file__))
GT_STEP_ROOT    = "/home/lukau/Desktop/abc_dataset/abc_0000_step_v00"
SAMPLE_DIR      = os.path.join(PROJECT_DIR, "sample_clouds")
BREP_DIR        = os.path.join(PROJECT_DIR, "output_brep")
MESH_DIR        = os.path.join(PROJECT_DIR, "output_bfs")
OVERSIZED_DIR   = os.path.join(PROJECT_DIR, "output_oversized")
UVALPHA_DIR     = os.path.join(PROJECT_DIR, "output_uvalpha")


def authenticate():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)


def get_or_create_folder(drive, name, parent_id):
    """Find or create a folder under parent_id. Returns folder ID."""
    q = (f"title='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    existing = drive.ListFile({"q": q}).GetList()
    if existing:
        return existing[0]["id"]
    folder = drive.CreateFile({
        "title": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [{"id": parent_id}],
    })
    folder.Upload()
    print(f"  Created folder: {name}")
    return folder["id"]


def upload_file(drive, local_path, parent_id, remote_name=None):
    """Upload a single file to a Drive folder."""
    name = remote_name or os.path.basename(local_path)
    f = drive.CreateFile({"title": name, "parents": [{"id": parent_id}]})
    f.SetContentFile(local_path)
    f.Upload()
    print(f"  Uploaded: {name}")


def collect_and_upload(drive, model_id, root_folder_id):
    """Collect all outputs for a model and upload to Drive."""
    # Strip abc_ prefix for GT lookup
    numeric_id = model_id.replace("abc_", "")

    model_folder_id = get_or_create_folder(drive, model_id, root_folder_id)

    # --- Ground truth STEP ---
    gt_dir = os.path.join(GT_STEP_ROOT, numeric_id)
    if os.path.isdir(gt_dir):
        gt_files = _glob.glob(os.path.join(gt_dir, "*.step"))
        if gt_files:
            upload_file(drive, gt_files[0], model_folder_id, "ground_truth.step")
    else:
        print(f"  No ground truth STEP for {model_id} — skipping")

    # --- Per-part point clouds ---
    sample_dir = os.path.join(SAMPLE_DIR, model_id)
    if os.path.isdir(sample_dir):
        xyzc_files = sorted(_glob.glob(os.path.join(sample_dir, "*.xyzc")),
                            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        if xyzc_files:
            pc_folder_id = get_or_create_folder(drive, "point_clouds", model_folder_id)
            for f in xyzc_files:
                upload_file(drive, f, pc_folder_id)

    # --- Per-part outputs ---
    brep_model_dir = os.path.join(BREP_DIR, model_id)
    mesh_model_dir = os.path.join(MESH_DIR, model_id)
    oversized_model_dir = os.path.join(OVERSIZED_DIR, model_id)
    uvalpha_model_dir = os.path.join(UVALPHA_DIR, model_id)

    # Discover parts from sample_clouds
    if not os.path.isdir(sample_dir):
        print(f"  No sample_clouds/{model_id}/ — cannot determine parts")
        return

    part_indices = sorted(
        int(os.path.splitext(os.path.basename(p))[0])
        for p in _glob.glob(os.path.join(sample_dir, "*.xyzc"))
    )

    for pi in part_indices:
        part_folder_id = get_or_create_folder(drive, f"part_{pi}", model_folder_id)

        # BRep STEP
        brep_step = os.path.join(brep_model_dir, f"part_{pi}", f"part_{pi}.step")
        if os.path.exists(brep_step):
            upload_file(drive, brep_step, part_folder_id, "brep.step")

        # Oversized STEP
        oversized_step = os.path.join(oversized_model_dir, f"part_{pi}.step")
        if os.path.exists(oversized_step):
            upload_file(drive, oversized_step, part_folder_id, "oversized.step")

        # UV alpha STEP
        uvalpha_step = os.path.join(uvalpha_model_dir, f"part_{pi}.step")
        if os.path.exists(uvalpha_step):
            upload_file(drive, uvalpha_step, part_folder_id, "uv_alpha.step")

        # Trimmed mesh (denormalized)
        trimmed_stl = os.path.join(mesh_model_dir, f"part_{pi}", "trimmed_denorm.stl")
        if os.path.exists(trimmed_stl):
            upload_file(drive, trimmed_stl, part_folder_id, "trimmed_mesh.stl")

    print(f"\nDone: {model_id}")


def report_part_counts(drive, root_folder_id):
    """Count total parts across all models based on .xyzc files on Google Drive."""
    # List model folders under root
    q = (f"'{root_folder_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    model_folders = drive.ListFile({"q": q}).GetList()
    model_folders.sort(key=lambda f: f["title"])

    total_parts = 0
    for mf in model_folders:
        # Look for point_clouds subfolder
        q_pc = (f"'{mf['id']}' in parents and title='point_clouds' "
                f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
        pc_folders = drive.ListFile({"q": q_pc}).GetList()
        if not pc_folders:
            continue
        # Count .xyzc files inside point_clouds
        q_xyzc = (f"'{pc_folders[0]['id']}' in parents "
                  f"and title contains '.xyzc' and trashed=false")
        xyzc_files = drive.ListFile({"q": q_xyzc}).GetList()
        n_parts = len(xyzc_files)
        if n_parts > 0:
            print(f"  {mf['title']}: {n_parts} parts")
            total_parts += n_parts

    print(f"\nTotal: {total_parts} parts across {len(model_folders)} models")


def main():
    parser = argparse.ArgumentParser(
        description="Collect and upload model results to Google Drive",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_id", type=str,
                        help="Model ID (e.g. abc_00000005)")
    parser.add_argument("--root_folder_id", type=str,
                        help="Google Drive folder ID for the upload root "
                             "(from the folder URL: drive.google.com/drive/folders/<ID>)",
                             default="1KV6o_6SCuOk9NGAx1fuYll4YMgYrQAJ_")
    parser.add_argument("--count-parts", action="store_true",
                        help="Report per-model part counts and exit")
    args = parser.parse_args()

    drive = authenticate()

    if args.count_parts:
        report_part_counts(drive, args.root_folder_id)
        return

    if not args.model_id:
        parser.error("--model_id is required when uploading")
    collect_and_upload(drive, args.model_id, args.root_folder_id)

    print("\n--- Cleanup reminder ---")
    print("When done uploading all models, delete these files:")
    print(f"  rm {os.path.join(PROJECT_DIR, 'credentials.json')}")
    print(f"  rm {os.path.join(PROJECT_DIR, 'client_secrets.json')}")


if __name__ == "__main__":
    main()
