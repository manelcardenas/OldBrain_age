import numpy as np
#from scipy.stats import norm
from load_female_data import female_data
import nibabel as nib
import os

dir_test_brain_age = "/home/usuaris/imatge/joan.manel.cardenas/brainAge"

def crop_center(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example: 
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    print(in_sp)
    nd = np.ndim(data)
    print(nd)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop


def crop_center_all(female_files_info, out_sp):
    """
    Crop center part of volume data for all subjects.
    :param female_files_info: List of tuples containing subject IDs and volume data.
    :param out_sp: Output spatial dimensions (out_sp < in_sp).
    :return: List of tuples containing subject IDs and cropped volume data.
    """
    cropped_data_info = []

    for subject_id, data in female_files_info:
        cropped_data = crop_center(data, out_sp)
        cropped_data_info.append((subject_id, cropped_data))

    return cropped_data_info


female_files_info = female_data()
out_sp = (160, 192, 160)  # Definir las dimensiones deseadas para el recorte

cropped_data_info = crop_center_all(female_files_info, out_sp)


for subject_id, cropped_data in cropped_data_info:
    # Crear un objeto Nifti1Image
    nifti_img = nib.Nifti1Image(cropped_data.squeeze(), np.eye(4))
    
    # Nombre del archivo .nii
    nifti_filename = os.path.join(dir_test_brain_age, f"{subject_id}_cropped_brain.nii")
    
    # Guardar la imagen en formato .nii
    nib.save(nifti_img, nifti_filename)


