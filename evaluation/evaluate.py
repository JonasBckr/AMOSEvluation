import miscnn
from miscnn.data_loading.interfaces import NIFTI_interface
from miscnn import Data_IO, Preprocessor, Data_Augmentation, Neural_Network
from miscnn.neural_network.architecture.unet.standard import Architecture

from miscnn.processing.subfunctions import Normalization, Clipping, Resampling
from miscnn.processing.subfunctions import Resize
import numpy as np
import pandas as pd


def create_predictions(data_path, model_path, data_io, model_type):
    print('Predicting segmentation for ' + str(len(data_io.get_indiceslist())) + ' images')
    pp = init_preprocessor(data_path, data_io, model_type)
    model = load_model(model_path, pp, model_type)
    #stores the predictions into a prediction directory in the directory of the script 
    sample_list = data_io.get_indiceslist()
    sample_list.sort()
    pred = model.predict(sample_list)
        
    

def load_model(model_path, preprocessor, model_type):
    if(model_type == '3D'):
        unet_standard = Architecture()
        model = miscnn.Neural_Network(preprocessor=preprocessor, architecture=unet_standard)
    elif(model_type):
        model = miscnn.Neural_Network(preprocessor=preprocessor)
    
    model.load(model_path)
    return model

def init_preprocessor(data_path, data_io, model_type):
    # Create a Preprocessor instance to configure how to preprocess the data into batches

    if(model_type == '3D'):
        # Create a resampling Subfunction to voxel spacing 1.58 x 1.58 x 2.70
        sf_resample = Resampling((1.58, 1.58, 2.70))
        # Create a pixel value normalization Subfunction for z-score scaling
        sf_zscore = Normalization(mode='z-score')
        # Assemble Subfunction classes into a list
        sf = [sf_resample, sf_zscore]
        pp = Preprocessor(data_io, batch_size=4, subfunctions=sf,
                prepare_subfunctions=True, prepare_batches=False, 
                analysis='patchwise-crop', patch_shape=(128, 128, 128))
        pp.patchwise_overlap = (64, 64, 64) #nur für prediction
    elif(model_type == '2D'):
        # want a specific shape (e.g. DenseNet for classification)
        sf_resize = Resize(new_shape=(224, 224))
        # Create a pixel value normalization Subfunction for z-score scaling
        sf_zscore = Normalization(mode="z-score")

        # Assemble Subfunction classes into a list
        sf = [sf_resize]

        # Create and configure the Preprocessor class
        pp = Preprocessor(data_io, data_aug=None, batch_size=1, subfunctions=sf,
                        prepare_subfunctions=True, prepare_batches=False, analysis="fullimage")

    return pp

def create_folder_structure(prediction_path, test_path):
    # create a directory for each case
    pass

def calc_scores(sample_index, data_io):
    sample = data_io.sample_loader(sample_index, load_seg=True, load_pred=True)
    image = sample.img_data
    truth = sample.seg_data
    pred = sample.pred_data

    scores = {
        'sample': sample_index,

    }
    dice = calc_DSC(truth, pred, 16)
    iou = calc_IoU(truth, pred, 16)
    sens = calc_Sensitivity(truth, pred, 16)
    spec = calc_Specificity(truth, pred, 16)
    acc = calc_Accuracy(truth, pred, 16)
    pre = calc_Precision(truth, pred, 16)
    for i in range(16):
        scores["dice" + str(i)] = dice[i]
        scores["iou" + str(i)] = iou[i]
        scores["sensitivity" + str(i)] = sens[i]
        scores["specificity" + str(i)] = spec[i]
        scores["accuracy" + str(i)] = acc[i]
        scores["precision" + str(i)] = pre[i]
    return scores

# from https://github.com/frankkramer-lab/covid19.MIScnn/blob/master/scripts/run_evaluation.py
#-----------------------------------------------------#
#                  Score Calculations                 #
#-----------------------------------------------------#
def calc_DSC(truth, pred, classes):
    dice_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)

            # Calculate Dice
            dice = 2*np.logical_and(pd, gt).sum() / (pd.sum() + gt.sum())
            dice_scores.append(dice)
        except ZeroDivisionError:
            dice_scores.append(0.0)
    # Return computed Dice Similarity Coefficients
    return dice_scores

def calc_IoU(truth, pred, classes):
    iou_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)
            # Calculate iou
            iou = np.logical_and(pd, gt).sum() / (pd.sum() + gt.sum() - np.logical_and(pd, gt).sum())
            iou_scores.append(iou)
        except ZeroDivisionError:
            iou_scores.append(0.0)
    # Return computed IoU
    return iou_scores

def calc_Sensitivity(truth, pred, classes):
    sens_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)
            # Calculate sensitivity
            sens = np.logical_and(pd, gt).sum() / gt.sum()
            sens_scores.append(sens)
        except ZeroDivisionError:
            sens_scores.append(0.0)
    # Return computed sensitivity scores
    return sens_scores

def calc_Specificity(truth, pred, classes):
    spec_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            not_gt = np.logical_not(np.equal(truth, i))
            not_pd = np.logical_not(np.equal(pred, i))
            # Calculate specificity
            spec = np.logical_and(not_pd, not_gt).sum() / (not_gt).sum()
            spec_scores.append(spec)
        except ZeroDivisionError:
            spec_scores.append(0.0)
    # Return computed specificity scores
    return spec_scores

def calc_Accuracy(truth, pred, classes):
    acc_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)
            not_gt = np.logical_not(np.equal(truth, i))
            not_pd = np.logical_not(np.equal(pred, i))
            # Calculate accuracy
            acc = (np.logical_and(pd, gt).sum() + \
                   np.logical_and(not_pd, not_gt).sum()) /  gt.size
            acc_scores.append(acc)
        except ZeroDivisionError:
            acc_scores.append(0.0)
    # Return computed accuracy scores
    return acc_scores

def calc_Precision(truth, pred, classes):
    prec_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)
            # Calculate precision
            prec = np.logical_and(pd, gt).sum() / pd.sum()
            prec_scores.append(prec)
        except ZeroDivisionError:
            prec_scores.append(0.0)
    # Return computed precision scores
    return prec_scores

def calc_average_scores(csv_path):
    scores = pd.read_csv(csv_path)
    scores = scores.dropna()
    dice_sum = 0
    iou_sum = 0
    specificity_sum = 0
    sensitivity_sum = 0
    accuracy_sum = 0
    precision_sum = 0
    for i in range(16):
        dice_sum = dice_sum + scores['dice' + str(i)]
        iou_sum = iou_sum + scores['iou' + str(i)]
        specificity_sum = specificity_sum + scores['sensitivity' + str(i)]
        sensitivity_sum = sensitivity_sum + scores['specificity' + str(i)]
        accuracy_sum = accuracy_sum + scores['accuracy' + str(i)]
        precision_sum = accuracy_sum + scores['precision' + str(i)]


    scores['mean_dice'] = dice_sum / 16
    scores['mean_iou'] = iou_sum / 16
    scores['mean_sensitivity'] = sensitivity_sum / 16
    scores['mean_specificity'] = specificity_sum / 16
    scores['mean_accuracy'] = accuracy_sum / 16
    scores['mean_precision'] = precision_sum / 16
    dice_sum = 0
    iou_sum = 0
    specificity_sum = 0
    sensitivity_sum = 0
    accuracy_sum = 0
    precision_sum = 0
    return scores


def evaluate_miscnn3D(data_path, model_path, output_path):
    interface = NIFTI_interface(pattern='amos_0[0-9][0-9][0-9]', channels=1, classes=16)
    data_io = miscnn.Data_IO(interface, data_path)
    sample_list = data_io.get_indiceslist()
    sample_list.sort()
    # only needs to be run once for each sample
    model_path ='/data/miscnn_models/model02092D/model.best.hdf5'
    #create_predictions(data_path, model_path, data_io, '3D')

    scores = pd.DataFrame()
    for index in sample_list:
        print("Processing: " + index)
        scores = scores.append(calc_scores(index, data_io), ignore_index=True)
        scores.to_csv(output_path + '/scores.csv')
        
    new_scores = calc_average_scores(output_path + '/scores.csv')
    new_scores.to_csv(output_path + '/scores.csv')

def evaluate_miscnn2D(data_path, model_path, output_path):
    interface = NIFTIslicer_interface(pattern='amos_0[0-9][0-9][0-9]', channels=1, classes=16)
    data_io = miscnn.Data_IO(interface, data_path)
    sample_list = data_io.get_indiceslist()
    sample_list.sort()
    # only needs to be run once for each sample
    model_path ='/data/miscnn_models/model02092D/model.best.hdf5'
    create_predictions(data_path, model_path, data_io, '2D')

    scores = pd.DataFrame()
    for index in sample_list:
        print("Processing: " + index)
        scores = scores.append(calc_scores(index, data_io), ignore_index=True)
        scores.to_csv(output_path + '/scores.csv')
        
    new_scores = calc_average_scores(output_path + '/scores.csv')
    new_scores.to_csv(output_path + '/scores.csv')

def evaluate_nnunet(data_path, output_path):
    interface = NIFTI_interface(pattern='amos_0[0-9][0-9][0-9]', channels=1, classes=16)
    data_io = miscnn.Data_IO(interface, data_path)
    sample_list = data_io.get_indiceslist()

    scores = pd.DataFrame()
    for index in sample_list:
        print("Processing: " + index)
        scores = scores.append(calc_scores(index, data_io), ignore_index=True)
        scores.to_csv(data_path + '/scores.csv')
        

    new_scores = calc_average_scores(data_path + '/scores.csv')
    new_scores.to_csv(data_path + '/scores.csv')


'''data_path = '/data/MISCNN_raw/Va'
# for 3D
 interface = NIFTI_interface(pattern='amos_0[0-9][0-9][0-9]', channels=1, classes=16)
# for 2D
#from nifti_slicer_io import NIFTIslicer_interface
#interface = NIFTIslicer_interface(pattern='amos_0[0-9][0-9][0-9]', channels=1, classes=16)


data_io = miscnn.Data_IO(interface, data_path)
sample_list = data_io.get_indiceslist()
sample_list.sort()

# only needs to be run once for each sample
create_predictions('/data/MISCNN_raw/Va', '/data/miscnn_models/model02092D/model.best.hdf5', data_io, '2D')

scores = pd.DataFrame()
for index in sample_list:
    print("Processing: " + index)
    scores = scores.append(calc_scores(index), ignore_index=True)
    scores.to_csv('/data/miscnn_models/model02092D/scores.csv')
    

new_scores = calc_average_scores('/data/miscnn_models/model02092D/scores.csv')
new_scores.to_csv('/data/miscnn_models/model02092D/scores.csv')
'''


# evaluate_miscnn2D('/data/MISCNN_raw/Va', '/data/miscnn_models/model02092D/model.latest.hdf5', '/data/miscnn_models/model02092D')
evaluate_miscnn3D('/data/MISCNN_raw/Va', '/data/miscnn_models/model0407/model.latest.hdf5', '/data/miscnn_models/model0407')
#evaluate_nnunet('/data/scripts/nnupredictions', '/data/scripts/nnupredictions')



