for TRAIN in /data/nnUNet_raw/Dataset001_Amos/imagesTr/*; do 
    NAME=$(echo $TRAIN | cut -d . -f 1)
    echo "${TRAIN/0001}";
    mv ${TRAIN} "${NAME}_0000.nii.gz"
    mv ${TRAIN} "${NAME}_0000.nii.gz"
done

for VAL in /data/nnUNet_raw/Dataset001_Amos/imagesVa/*; do 
    NAME=$(echo $VAL | cut -d . -f 1)
    echo "${NAME}";
    mv "${VAL}" "${VAL//_0001/_0000}";
    mv ${VAL} "${NAME}_0000.nii.gz"
done

for TEST in /data/nnUNet_raw/Dataset001_Amos/imagesTs/*; do 
    NAME=$(echo $TEST | cut -d . -f 1)
    echo "${NAME}";
    mv ${TEST} "${NAME}_0000.nii.gz"
done