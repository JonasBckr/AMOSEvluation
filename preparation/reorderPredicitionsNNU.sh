PREDICTIONS_DIR="/data/scripts/nnupredictions/"

for TEST in /data/nnUNet_raw/Dataset001_Amos/imagesTs/*.nii.gz; do 
    NAME=$(echo $TEST | cut -d / -f 6 | cut -d . -f 1 | cut -d _ -f-1,2)
    # echo ${NAME}
    mkdir "${PREDICTIONS_DIR}/${NAME}";
    # copy the original image to the folder
    cp "${TEST}" "${PREDICTIONS_DIR}/${NAME}/";
    # copy the prediction
    cp "${PREDICTIONS_DIR}${NAME}.nii.gz" "${PREDICTIONS_DIR}/${NAME}/"

    #rename the segmentation and image to the miscnn format (using miscnn interface for evaluation)
    mv "${PREDICTIONS_DIR}${NAME}/${NAME}.nii.gz" "${PREDICTIONS_DIR}${NAME}/segmentation.nii.gz"
    mv "${PREDICTIONS_DIR}${NAME}/${NAME}_0000.nii.gz" "${PREDICTIONS_DIR}${NAME}/imaging.nii.gz"
done   