import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

patients = ["BRATS_457", "BRATS_306", "BRATS_206"]
modalities = ["T1", "T1ce", "T2", "FLAIR"]

for patient in patients:

    directory = f"processed/brats_test/{patient}"

    mask = nib.load(
        f"{directory}/{patient}_tumor_mask.nii.gz"
    ).get_fdata()

    z = mask.shape[2] // 2

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    for i, modality in enumerate(modalities):

        image = nib.load(
            f"{directory}/{patient}_{modality}_normalized.nii.gz"
        ).get_fdata()

        ax[i].imshow(
            image[:, :, z].T,
            cmap="gray",
            origin="lower"
        )

        tumor = np.ma.masked_where(
            mask[:, :, z].T == 0,
            mask[:, :, z].T
        )

        ax[i].imshow(
            tumor,
            cmap="jet",
            alpha=0.35,
            origin="lower"
        )

        ax[i].set_title(modality)
        ax[i].axis("off")

    plt.suptitle(
        f"{patient} — Final Preprocessing QC"
    )

    plt.tight_layout()
    plt.show()

    print(
        patient,
        "MRI shape:",
        image.shape,
        "Mask shape:",
        mask.shape,
        "Labels:",
        np.unique(mask)
    )

