import os
import time
import subprocess
import nibabel as nib
import numpy as np
import SimpleITK as sitk

DATASET = "Task01_BrainTumour"
OUTPUT = "processed/full"

MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]

# Read all patients from the split files
patients = []

for split in ["train", "val", "test"]:
    with open(f"splits/{split}.txt") as f:
        patients.extend([x.strip() for x in f if x.strip()])


def extract_modalities(patient, outdir):

    input_file = f"{DATASET}/imagesTr/{patient}.nii.gz"

    img = nib.load(input_file)
    data = img.get_fdata()

    for i, modality in enumerate(MODALITIES):

        output = f"{outdir}/{patient}_{modality}.nii.gz"

        if os.path.exists(output):
            continue

        volume = data[:, :, :, i]

        new_img = nib.Nifti1Image(
            volume.astype(np.float32),
            img.affine,
            img.header
        )

        nib.save(new_img, output)

        print("Extracted:", modality)


def skullstrip_flair(patient, outdir):

    flair = f"{outdir}/{patient}_FLAIR.nii.gz"
    brain = f"{outdir}/{patient}_FLAIR_brain.nii.gz"
    mask = f"{outdir}/{patient}_brain_mask.nii.gz"

    if os.path.exists(brain) and os.path.exists(mask):
        return

    print("Running SynthStrip on FLAIR...")

    command = [
        "podman", "run", "--rm",
        "-v", f"{os.path.abspath(outdir)}:/data:Z",
        "docker.io/freesurfer/synthstrip:latest",
        "-i", f"/data/{patient}_FLAIR.nii.gz",
        "-o", f"/data/{patient}_FLAIR_brain.nii.gz",
        "-m", f"/data/{patient}_brain_mask.nii.gz"
    ]

    subprocess.run(command, check=True)


def apply_mask(patient, modality, outdir):

    input_file = f"{outdir}/{patient}_{modality}.nii.gz"
    output_file = f"{outdir}/{patient}_{modality}_brain.nii.gz"
    mask_file = f"{outdir}/{patient}_brain_mask.nii.gz"

    if os.path.exists(output_file):
        return

    image = sitk.ReadImage(
        input_file,
        sitk.sitkFloat32
    )

    mask = sitk.ReadImage(
        mask_file,
        sitk.sitkUInt8
    )

    masked = sitk.Mask(image, mask)

    sitk.WriteImage(
        masked,
        output_file
    )


def n4_correct(patient, modality, outdir):

    input_file = f"{outdir}/{patient}_{modality}_brain.nii.gz"
    output_file = f"{outdir}/{patient}_{modality}_n4.nii.gz"

    if os.path.exists(output_file):
        return

    print(f"N4: {modality}")

    start = time.time()

    image = sitk.ReadImage(
        input_file,
        sitk.sitkFloat32
    )

    mask = sitk.Cast(
        image > 0,
        sitk.sitkUInt8
    )

    n4 = sitk.N4BiasFieldCorrectionImageFilter()

    n4.SetMaximumNumberOfIterations(
        [3, 3, 3]
    )

    corrected = n4.Execute(
        image,
        mask
    )

    sitk.WriteImage(
        corrected,
        output_file
    )

    print(
        f"N4 finished in {time.time() - start:.1f}s"
    )


def normalize(patient, modality, outdir):

    input_file = f"{outdir}/{patient}_{modality}_n4.nii.gz"
    output_file = f"{outdir}/{patient}_{modality}_normalized.nii.gz"

    if os.path.exists(output_file):
        return

    img = nib.load(input_file)

    data = img.get_fdata().astype(np.float32)

    brain = data[data > 0]

    p1 = np.percentile(brain, 1)
    p99 = np.percentile(brain, 99)

    clipped = np.clip(
        data,
        p1,
        p99
    )

    values = clipped[data > 0]

    mean = values.mean()
    std = values.std()

    normalized = np.zeros_like(data)

    normalized[data > 0] = (
        (values - mean) / std
    )

    nib.save(
        nib.Nifti1Image(
            normalized,
            img.affine,
            img.header
        ),
        output_file
    )


def copy_tumor_mask(patient, outdir):

    source = f"{DATASET}/labelsTr/{patient}.nii.gz"
    destination = f"{outdir}/{patient}_tumor_mask.nii.gz"

    if os.path.exists(destination):
        return

    img = nib.load(source)

    nib.save(
        img,
        destination
    )


# ==========================================================
# MAIN
# ==========================================================

print(f"Total patients: {len(patients)}")

for number, patient in enumerate(patients, 1):

    print("\n" + "=" * 70)
    print(f"[{number}/{len(patients)}] {patient}")
    print("=" * 70)

    start = time.time()

    outdir = f"{OUTPUT}/{patient}"

    os.makedirs(
        outdir,
        exist_ok=True
    )

    # 1. Extract T1/T1ce/T2/FLAIR
    extract_modalities(
        patient,
        outdir
    )

    # 2. SynthStrip FLAIR only
    skullstrip_flair(
        patient,
        outdir
    )

    # 3. Apply same aligned brain mask
    # 4. N4
    # 5. Normalize

    for modality in MODALITIES:

        apply_mask(
            patient,
            modality,
            outdir
        )

        n4_correct(
            patient,
            modality,
            outdir
        )

        normalize(
            patient,
            modality,
            outdir
        )

    # 6. Keep tumor ground truth untouched
    copy_tumor_mask(
        patient,
        outdir
    )

    elapsed = time.time() - start

    print(
        f"\nCompleted {patient} in "
        f"{elapsed / 60:.1f} minutes"
    )

print("\nFULL DATASET PREPROCESSING COMPLETE")
