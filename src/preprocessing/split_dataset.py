import os
import random

image_dir = "Task01_BrainTumour/imagesTr"
label_dir = "Task01_BrainTumour/labelsTr"

# Only use patients that have BOTH MRI and tumor mask
images = {
    f[:-7]
    for f in os.listdir(image_dir)
    if f.endswith(".nii.gz")
}

labels = {
    f[:-7]
    for f in os.listdir(label_dir)
    if f.endswith(".nii.gz")
}

patients = sorted(images & labels)

print("Total paired patients:", len(patients))

# Reproducible random split
random.seed(42)
random.shuffle(patients)

n = len(patients)

n_train = int(0.80 * n)
n_val = int(0.10 * n)

train = patients[:n_train]
val = patients[n_train:n_train + n_val]
test = patients[n_train + n_val:]

os.makedirs("splits", exist_ok=True)

with open("splits/train.txt", "w") as f:
    f.write("\n".join(train))

with open("splits/val.txt", "w") as f:
    f.write("\n".join(val))

with open("splits/test.txt", "w") as f:
    f.write("\n".join(test))

print("Training:", len(train))
print("Validation:", len(val))
print("Testing:", len(test))

print("\nSaved:")
print("splits/train.txt")
print("splits/val.txt")
print("splits/test.txt")
