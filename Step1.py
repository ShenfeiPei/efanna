data_path = "D:/DATA/"
funs_path = "D:/project/pyfuns/"
project_path = "D:/project/efanna_win/"
import sys
sys.path.append(funs_path)
import my2_0 as my
import numpy as np
import os

# names = ["Adience_29v2", "CACD_29v2", "CALFW_29v2", "CAS_PEAL_29v2", "CASIA_FaceV5_29v2", "CASIA_WebFace_29v2",
#          "CelebA_29v2", "CFP_29v2", "CK_29v2", "CK+_29v2", "CNBC_29v2", "CPLFW_29v2", "EYaleB_29v2", "face94_29v2",
#          "face95_29v2", "face96_29v2", "FERET_29v2", "grimace_29v2", "IMDB_29v2", "IMM_29v2", "jaffe_29v2",
#          "LFW_29v2", "MUCT_29v2", "PINS_29v2", "RaFD_29v2", "YaleFace_29v2", "YouTubeFace_29v2", "ORL_29v2"]

names = ["Adience_29v2", "CACD_29v2", "CAS_PEAL_29v2", "CASIA_WebFace_29v2", "CelebA_29v2"]
knn = 50
nTree = 8
depth = 6
ITER = 8
L = knn + 5
check = knn + 5
S = 30
exe_name = project_path + "samples/efanna_index_buildgraph"

data_i = 0
data_name = names[data_i]

for data_i, data_name in enumerate(names):
    X, y_true, N, dim, c_true = my.load_mat(data_path + "FaceData/" + data_name + ".mat")
    print(data_name, N, dim, c_true)

    Xbin_full_name = data_path + "bin/" + data_name + ".bin"
    if not os.path.exists(Xbin_full_name):
        X.astype(np.float32).tofile(Xbin_full_name)

    graph_full_name = data_path + "bin/" + data_name + "_" + str(knn) + ".graph"
    cmd = "{exe} {data_name} {data_full_name} {N} {dim} {gra} {nTree} {depth} {ITER} {L} {check} {knn} {S} > {out}.txt".format(
        exe=exe_name, data_name=data_name, data_full_name=Xbin_full_name, N=N, dim=dim, gra=graph_full_name,
        nTree=nTree, depth=depth, ITER=ITER, L=L, check=check, knn=knn-1, S=S, out=data_name+str(knn))

    with open(project_path + "cmd/" + data_name + ".bat", 'w') as f:
        f.write(cmd)
