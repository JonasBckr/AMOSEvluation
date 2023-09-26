# AMOSEvaluation
Scripts for the comparison of the MIScnn and nnU-Net medical image segmentation frameworks on the AMOS dataset

# Preparation
These scripts were used to move and rename directories of the AMOS dataset to be in the right format for the interfaces of the two frameworks.

# MIScnn
Contains the compilation and training of the 3D model. During this project I worked on a 2.5D configuration of MIScnn using the nifti_slicer interface. Since the code of the interface is bugged in the MIScnn repository I downloaded it and modified it. However the 2.5D model did not compile in time for the evaluation. To figure out the values of the resampling preprocessing function the identify_resamplingShape.py scripts was used.

# nnUnet
For the training of the model and creation of predictions the commands in the commands.sh files were used. The configuration of the hyperparameters that nnU-Net chose can be found in the .json files.

# Evaluation
For the evaluation the images were loaded using the MIScnn interface. The boxplots of the paper were created using the visualize_evaluation.py script.
