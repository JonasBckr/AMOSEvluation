for TRAIN in /data/nnUNet_raw/imagesTr/*; do 
    SAMPLEID=$(echo $TRAIN | cut -d _ -f 3 | cut -d . -f 1)
    echo "${SAMPLEID}"
    LABEL=$(echo /data/nnUNet_raw/labelsTr/amos_${SAMPLEID}*) ;
    echo "${LABEL}"
    mkdir "/data/MISCNN_raw/Tr/amos_${SAMPLEID}"

    cp $TRAIN /data/MISCNN_raw/Tr/amos_${SAMPLEID}/imaging.nii.gz
    cp $LABEL /data/MISCNN_raw/Tr/amos_${SAMPLEID}/segmentation.nii.gz
done

for VAL in /data/nnUNet_raw/imagesVa/*; do 
    SAMPLEID=$(echo $VAL | cut -d _ -f 3 | cut -d . -f 1)
    echo "${SAMPLEID}"
    LABEL=$(echo /data/nnUNet_raw/labelsVa/amos_${SAMPLEID}*) ;
    echo "${LABEL}"
    mkdir "/data/MISCNN_raw/Va/amos_${SAMPLEID}"

    cp $VAL /data/MISCNN_raw/Va/amos_${SAMPLEID}/imaging.nii.gz
    cp $LABEL /data/MISCNN_raw/Va/amos_${SAMPLEID}/segmentation.nii.gz
done