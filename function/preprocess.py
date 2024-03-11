import numpy as np
import matplotlib.pyplot as plt
import os


def processmgz(brains):
    # brains = []
    # files = glob(os.path.join(path_of_mgz, "*.mgz"))
    # for f in files:
    #     mgh_file = mgh.load(f)  # for each mgz file
    #     new_array = zoom(mgh_file.get_data(), (0.5, 0.5, 0.5))
    #     brains.append(new_array)

   # brains = np.asarray(brains)

    X = np.asarray(brains)
    coord = []
    for i in range(X.shape[0]):
        buf = X[i, :, :, :]
        print(buf.shape)
        buf = buf.reshape((128, 128, 128))
        xmin = 0
        xmax = 0
        ymin = 0
        ymax = 0
        zmin = 0
        zmax = 0
        for xm in range(0, 128):
            if np.sum(buf[xm, :, :]) > 50:
                xmin = xm
                break
        for xm in range(0, 128):
            if np.sum(buf[127 - xm, :, :]) > 50:
                xmax = 127 - xm
                break
        for ym in range(0, 128):
            if np.sum(buf[:, ym, :]) > 50:
                ymin = ym
                break
        for ym in range(0, 128):
            if np.sum(buf[:, 127 - ym, :]) > 50:
                ymax = 127 - ym
                break
        for zm in range(0, 128):
            if np.sum(buf[:, :, zm]) > 50:
                zmin = zm
                break
        for zm in range(0, 128):
            if np.sum(buf[:, :, 127 - zm]) > 50:
                zmax = 127 - zm
                break
        td = []
        td.append(abs(xmax + xmin) / 2)
        td.append(abs(ymax + ymin) / 2)
        td.append(abs(zmax + zmin) / 2)
        coord.append(td)

    x_range, y_range, z_range = 82, 86, 100
    data_new = []
    label = []
    id_list = []
    for i in range(X.shape[0]):
        buf = X[i, :, :, :]
        co = coord[i]
        buf = buf.reshape((128, 128, 128))
        if (
            co[0] > (x_range / 2)
            and co[0] < (128 - (x_range / 2))
            and co[1] > (y_range / 2)
            and co[1] < (128 - (y_range / 2))
            and co[2] > (z_range / 2)
            and co[2] < (128 - (z_range / 2))
        ):
            data_new.append(
                buf[
                    int(int(co[0]) - (x_range / 2)) : int(int(co[0]) + (x_range / 2)), #No tiene sentido, estas añadiendo redundancia.  co[0] es 89, co[1]=109.5 y co[2]=77 y x_range= 82, y_range=86, z_range=100. Sin embargo da igual. El resiltado siempre va a ser 82
                    int(int(co[1]) - (y_range / 2)) : int(int(co[1]) + (y_range / 2)),
                    int(int(co[2]) - (z_range / 2)) : int(int(co[2]) + (z_range / 2)),
                ]
            )
        else:
            coord[i][0] = (
                co[0]
                if co[0] in range(int(x_range / 2), int(128 - (x_range / 2)))
                else int(
                    min(
                        [x_range / 2, 128 - (x_range / 2)], key=lambda x: abs(x - co[0])
                    )
                )
            )
            coord[i][1] = (
                co[1]
                if co[1] in range(int(y_range / 2), int(128 - (y_range / 2)))
                else int(
                    min(
                        [y_range / 2, 128 - (y_range / 2)], key=lambda x: abs(x - co[1])
                    )
                )
            )
            coord[i][2] = (
                co[2]
                if co[2] in range(int(z_range / 2), int(128 - (z_range / 2)))
                else int(
                    min(
                        [z_range / 2, 128 - (z_range / 2)], key=lambda x: abs(x - co[2])
                    )
                )
            )
            data_new.append(
                buf[
                    int(int(coord[i][0]) - (x_range / 2)) : int(
                        int(coord[i][0]) + (x_range / 2)
                    ),
                    int(int(coord[i][1]) - (y_range / 2)) : int(
                        int(coord[i][1]) + (y_range / 2)
                    ),
                    int(int(coord[i][2]) - (z_range / 2)) : int(
                        int(coord[i][2]) + (z_range / 2)
                    ),
                ]
            )
    data_new = np.expand_dims(data_new, axis=4)  # reshape the brain MRI from 3D to 4D
    return data_new, coord


# if __name__ == '__main__':
#     tmp=np.zeros((82, 86, 100))
#     data = process_brain_mgz(tmp)
# return 0
