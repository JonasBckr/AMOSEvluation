# commands for training and inference
nohup nnUNetv2_train Dataset001_Amos 3d_lowres 0 
nohup nnUNetv2_train Dataset001_Amos 3d_lowres 1 
nohup nnUNetv2_train Dataset001_Amos 3d_lowres 2
nohup nnUNetv2_train Dataset001_Amos 3d_lowres 3
nohup nnUNetv2_train Dataset001_Amos 3d_lowres 4 
nohup nnUNetv2_find_best_configuration Dataset001_Amos -c 3d_lowres
nohup nnUNetv2_predict -i /data/nnUNet_raw/Dataset001_Amos/imagesTs -o /data/scripts/nnupredictions -d Dataset001_Amos -c 3d_lowres --save_probabilities
