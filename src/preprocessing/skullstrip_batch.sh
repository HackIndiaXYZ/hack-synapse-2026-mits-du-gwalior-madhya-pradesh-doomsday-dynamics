#!/bin/bash

PATIENTS=("BRATS_457" "BRATS_306" "BRATS_206")
MODALITIES=("T1" "T1ce" "T2" "FLAIR")

for PATIENT in "${PATIENTS[@]}"; do

    DIR="processed/brats_test/$PATIENT"

    echo "======================================"
    echo "Processing $PATIENT"
    echo "======================================"

    for MODALITY in "${MODALITIES[@]}"; do

        echo ""
        echo "Skull stripping $MODALITY..."

        podman run --rm \
            -v "$PWD/$DIR:/data:Z" \
            docker.io/freesurfer/synthstrip:latest \
            -i "/data/${PATIENT}_${MODALITY}.nii.gz" \
            -o "/data/${PATIENT}_${MODALITY}_brain.nii.gz" \
            -m "/data/${PATIENT}_${MODALITY}_brain_mask.nii.gz"

        if [ $? -ne 0 ]; then
            echo "ERROR: SynthStrip failed for $PATIENT $MODALITY"
            exit 1
        fi

    done
done

echo ""
echo "======================================"
echo "All skull stripping completed!"
echo "======================================"

