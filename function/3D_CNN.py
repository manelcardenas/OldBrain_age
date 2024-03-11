import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import nibabel as nib

sys.path.append("/home/usuaris/imatge/joan.manel.cardenas/brainAge/BA_estimator/function")

import pandas as pd
from preprocess_2 import processmgz
from load_female_data import female_data
import savepath_updated
import repadding
#from pandas import read_csv
#import tensorflow as tf
#from tensorflow import keras
import modelcnn
import saliencymaps
#import multiprocessing as mp



# path to the folder
dir_test_brain_age = "/home/usuaris/imatge/joan.manel.cardenas/brainAge/BA_estimator/woman_data"

# path to the h5 file
path_to_model_weights = os.path.join(
    "/home/usuaris/imatge/joan.manel.cardenas/brainAge/BA_estimator/function/model/saved-model-317.h5"
)


def mainfunction(dir_test_brain_age, path_to_model_weights):
    female_info_list = []
    subj_id = []
    brains_tmp = []
    female_info_list = female_data()
    # Obtener los IDs de los sujetos
    subj_id = [file_info[0] for file_info in female_info_list]

    # Obtener los datos de las imágenes MRI
    brains_tmp = [file_info[1] for file_info in female_info_list]
    
    brains, coordinates = processmgz(brains_tmp)
    (
        brain_save_path,
        coordinates_save_path,
        coordinates_save_path_csv,
    ) = savepath_updated.filename_brainnpy(dir_test_brain_age)

    # save the cropped brains
    np.save(brain_save_path, brains)

    # save information for repadding.
    np.save(coordinates_save_path, coordinates)
    df_coordinates = pd.DataFrame(coordinates)
    df_coordinates.columns = ["x", "y", "z"]
    df_coordinates["subj_id"] = subj_id
    df_coordinates.to_csv(coordinates_save_path_csv)

    ##TESTING
    # Guardar las imágenes en formato .nii después de las modificaciones
    for i, brain in enumerate(brains):
        nifti_img = nib.Nifti1Image(brain.squeeze(), np.eye(4))  # Crear un objeto Nifti1Image
        nifti_filename = os.path.join(dir_test_brain_age, f"{subj_id[i]}_brain.nii")  # Nombre del archivo .nii
        nib.save(nifti_img, nifti_filename)  # Guardar la imagen en formato .nii


    # Build model. 
    model = modelcnn.get_model(width=82, height=86, depth=100)

    # Load best weights.
    model.load_weights(path_to_model_weights)

    # make brain age predictions
    predictions = model.predict(brains) + 22

    # save the prediction results. Of note, the input subject ids should be saved as csv.
    
    BA_save_path_csv = savepath_updated.filename_pred(dir_test_brain_age)
    df_BA = pd.DataFrame(predictions)
    df_BA.columns = ["BA"]
    df_BA["subj_id"] = subj_id
    df_BA.to_csv(BA_save_path_csv)

    # Generate saliency mapes
    saliency_map_list = saliencymaps.smap(model, brains)

    # Repadding the saliency maps
    coordinates = np.load(coordinates_save_path)
    repadded_saliency_maps = repadding.repadding(saliency_map_list, coordinates)

    # Save the saliency maps
    smap_path = savepath_updated.filename_smap(dir_test_brain_age)
    np.save(smap_path, repadded_saliency_maps)
    return df_BA, repadded_saliency_maps



predicted_BA, repadded_saliency_maps = mainfunction(dir_test_brain_age, path_to_model_weights)

print(predicted_BA)
