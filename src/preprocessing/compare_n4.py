import nibabel as nib
import matplotlib.pyplot as plt

original = nib.load(
    "processed/brats_457/BRATS_457_T1_n4.nii.gz"
).get_fdata()

fast = nib.load(
    "processed/brats_457/BRATS_457_T1_n4_fast.nii.gz"
).get_fdata()

z = original.shape[2] // 2

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(original[:, :, z].T, cmap="gray", origin="lower")
ax[0].set_title("Original N4")
ax[0].axis("off")

ax[1].imshow(fast[:, :, z].T, cmap="gray", origin="lower")
ax[1].set_title("Fast N4")
ax[1].axis("off")

plt.tight_layout()
plt.show()

