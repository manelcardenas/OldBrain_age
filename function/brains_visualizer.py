import numpy as np
import os
import nibabel as nib

# Ruta donde se encuentran los archivos .npy
directory = "/home/usuaris/imatge/joan.manel.cardenas/brainAge"

def visualize_npy_data(directory):
    # Obtener la lista de archivos .npy en el directorio
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]

    if not npy_files:
        print("No se encontraron archivos .npy en el directorio especificado.")
        return

    # Iterar sobre cada archivo .npy y crear archivos .nii
    for npy_file in npy_files:
        # Construir la ruta completa al archivo .npy
        npy_path = os.path.join(directory, npy_file)
        
        # Cargar los datos desde el archivo .npy
        loaded_data = np.load(npy_path)
        
        # Crear un objeto NIfTI1 con los datos cargados
        nifti_img = nib.Nifti1Image(loaded_data, np.eye(4))  # Asigna la matriz de identidad como affine
        
        # Guardar la imagen NIfTI en formato .nii
        nii_filename = os.path.splitext(npy_file)[0] + ".nii"  # Cambia la extensión del archivo a .nii
        nii_path = os.path.join(directory, nii_filename)
        nib.save(nifti_img, nii_path)
        
        print(f"Archivo .nii creado: {nii_path}")

# Llamar a la función para crear los archivos .nii
visualize_npy_data(directory)

