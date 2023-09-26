# https://github.com/frankkramer-lab/MIScnn/blob/master/tutorials/NIfTIslicer_interface.py

# Import the MIScnn module
import miscnn
import tensorflow as tf
import os
import numpy as np
import random

from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, \
                                       EarlyStopping, CSVLogger, ModelCheckpoint


# Create a Data I/O interface
from nifti_slicer_io import NIFTIslicer_interface
# Nifti slicer selber einbinden (überarbeiten wie es in nift_io aussehen müsste)



# Set the random seed for stable results
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

# start time
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%d - %H:%M:%S")
print("Start Time =", current_time)

# here we use the NIFITslicer_interface instead of the normal 3D NIFTI Interface
interface = NIFTIslicer_interface(pattern="amos_0[0-9][0-9][0-9]", channels=1, classes=16)

# Initialize data path and create the Data I/O instance
data_path = "/data/MISCNN_raw/Tr"
fold_subdir = "/data/miscnn_models/model02092D"
data_io = miscnn.Data_IO(interface, data_path)
print(data_io.get_indiceslist())



from miscnn import Data_IO, Preprocessor, Data_Augmentation, Neural_Network
from miscnn.processing.subfunctions import Normalization, Resize

# no data augmentation as this increases runtime drastically > 60h
#data_aug = Data_Augmentation(cycles=1, scaling=True, rotations=True,
#                             elastic_deform=True, mirror=True,
#                             brightness=True, contrast=True, gamma=True,
#                             gaussian_noise=True)




# Specify subfunctions for preprocessing
## Here we are using the Resize subfunctions due to many 2D models
## want a specific shape (e.g. DenseNet for classification)
sf_resize = Resize(new_shape=(224, 224))
# Create a pixel value normalization Subfunction for z-score scaling
sf_zscore = Normalization(mode="z-score")

# Assemble Subfunction classes into a list
sf = [sf_resize]

# Create and configure the Preprocessor class
pp = Preprocessor(data_io, data_aug=None, batch_size=1, subfunctions=sf,
                  prepare_subfunctions=True, prepare_batches=False, analysis="fullimage")
## We are using fullimage analysis due to a 2D image can easily fit completely
## in our GPU

# Create a deep learning neural network model
model = miscnn.Neural_Network(preprocessor=pp)


# Define Callbacks
cb_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=15,
                          verbose=1, mode='min', min_delta=0.0001, cooldown=1,
                          min_lr=0.00001)
cb_es = EarlyStopping(monitor="loss", patience=25)
cb_tb = TensorBoard(log_dir=os.path.join(fold_subdir, "tensorboard"),
                    histogram_freq=0, write_graph=True, write_images=True)
cb_cl = CSVLogger(os.path.join(fold_subdir, "logs.csv"), separator=',',
                  append=True)
cb_mc = ModelCheckpoint(os.path.join(fold_subdir, "model.best.hdf5"),
                        monitor="loss", verbose=1,
                        save_best_only=True, mode="min")


# Training the model
from miscnn.evaluation import split_validation, cross_validation
sample_list = data_io.get_indiceslist()

now = datetime.now()

current_time = now.strftime("%d - %H:%M:%S")
print("Starting Training =", current_time)


# test code to remove
#cross_validation(sample_list, model, epochs=10, iterations=20, k_fold=3,
#                 draw_figures=True, evaluation_path="/data/miscnn_models/model02092D", run_detailed_evaluation=True, callbacks=[cb_lr, cb_es, cb_tb, cb_cl, cb_mc])

cross_validation(sample_list, model, epochs=500, iterations=150, k_fold=3,
                 draw_figures=True, evaluation_path="/data/miscnn_models/model02092D", run_detailed_evaluation=True, callbacks=[cb_lr, cb_es, cb_tb, cb_cl, cb_mc])


# Dump latest model to disk
model.dump(os.path.join(fold_subdir, "model.latest.hdf5"))

# stop time
now = datetime.now()

current_time = now.strftime("%d - %H:%M:%S")
print("Stop Time =", current_time)