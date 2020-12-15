data_path = "D:/DATA/"
funs_path = "D:/project/pyfuns/"
project_path = "D:/project/efanna_win/"
import time
import sys
sys.path.append(funs_path)
import my2_0 as my
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as EuDist2
import os

# names = ["Adience_29v2", "CACD_29v2", "CALFW_29v2", "CAS_PEAL_29v2", "CASIA_FaceV5_29v2", "CASIA_WebFace_29v2",
#          "CelebA_29v2", "CFP_29v2", "CK_29v2", "CK+_29v2", "CNBC_29v2", "CPLFW_29v2", "EYaleB_29v2", "face94_29v2",
#          "face95_29v2", "face96_29v2", "FERET_29v2", "grimace_29v2", "IMDB_29v2", "IMM_29v2", "jaffe_29v2",
#          "LFW_29v2", "MUCT_29v2", "PINS_29v2", "RaFD_29v2", "YaleFace_29v2", "YouTubeFace_29v2", "ORL_29v2"]

names = ["Adience_29v2", "CACD_29v2", "CAS_PEAL_29v2", "CASIA_WebFace_29v2", "CelebA_29v2", "IMDB_29v2", "YouTubeFace_29v2"]

data_i = 0
data_name = names[data_i]
knn = 50
for data_i, data_name in enumerate(names):
    X, y_true, N, dim, c_true = my.load_mat(data_path + "FaceData/" + data_name + ".mat")
    print(data_name, N, dim, c_true)

    graph_full_name = data_path + "bin/" + data_name + "_" + str(knn) + ".graph"
    e2_full_name = data_path + "bin/" + data_name + "_" + str(knn) + ".e2"

    if os.path.exists(e2_full_name):
        continue

    NN = np.fromfile(graph_full_name, dtype=np.int32)
    NN = NN.reshape(N, -1)

    if np.max(NN) >= N:
        print("Bad graph file, removed")
        os.system("rm {}".format(graph_full_name))
        continue

    NN[:, 0] = np.array(range(N), dtype=np.int32)
    knn = NN.shape[1]
    NND = np.zeros((N, knn))

    t1 = time.time()
    x_norm = np.sum(X**2, axis=1)

    for i in range(N):
        NND[i, :] = EuDist2(X[i, :].reshape(1, -1), X[NN[i, :], :], squared=True, X_norm_squared=x_norm[i:(i+1)].reshape(1, -1), Y_norm_squared=x_norm[NN[i, :]])
        # NND[i, :] = my.EuDist2(X[i, :].reshape(1, -1), X[NN[i, :], :], squared=True)
    t2 = time.time() - t1
    print(t2)
    NN.astype(np.int32).tofile(graph_full_name)
    NND.astype(np.float64).tofile(e2_full_name)
