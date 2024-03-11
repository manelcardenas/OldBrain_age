import numpy as np
from scipy.ndimage import zoom

def processmgz(brains):
    x_range = 82
    y_range = 86
    z_range = 100
    X = np.asarray(brains)
    coord = []
    for i in range(X.shape[0]):
        buf = X[i, :, :, :]
        xmin, xmax, ymin, ymax, zmin, zmax = 0, 0, 0, 0, 0, 0
        for xm in range(0, buf.shape[0]):
            if np.sum(buf[xm, :, :]) > 50:
                xmin = xm
                break
        for xm in range(0, buf.shape[0]):
            if np.sum(buf[buf.shape[0] - xm - 1, :, :]) > 50:
                xmax = buf.shape[0] - xm - 1
                break
        for ym in range(0, buf.shape[1]):
            if np.sum(buf[:, ym, :]) > 50:
                ymin = ym
                break
        for ym in range(0, buf.shape[1]):
            if np.sum(buf[:, buf.shape[1] - ym - 1, :]) > 50:
                ymax = buf.shape[1] - ym - 1
                break
        for zm in range(0, buf.shape[2]):
            if np.sum(buf[:, :, zm]) > 50:
                zmin = zm
                break
        for zm in range(0, buf.shape[2]):
            if np.sum(buf[:, :, buf.shape[2] - zm - 1]) > 50:
                zmax = buf.shape[2] - zm - 1
                break
        td = [(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2]
        coord.append(td)
    
    # Modificación para redimensionar las imágenes
    data_new = []
    for brain in X:
        # Calculamos los factores de redimensionamiento para cada eje
        x_factor = x_range / brain.shape[0]
        y_factor = y_range / brain.shape[1]
        z_factor = z_range / brain.shape[2]

        # Redimensionamos la imagen cerebral
        resized_brain = zoom(brain, (x_factor, y_factor, z_factor), order=1)  # order=1 para interpolación bilineal
        data_new.append(resized_brain)

    # Convertimos la lista en un array de NumPy y añadimos una nueva dimensión
    data_new = np.expand_dims(np.array(data_new), axis=4)
    
    return data_new, coord

