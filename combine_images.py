import numpy as np
import os
import matplotlib.pyplot as plt


data_dir = "galaxy_files"


imgs = []
ids = []
for i, file in enumerate(os.listdir(data_dir)):
    print(f"\r{file} {i}", end="")
    if file.endswith(".jpg"):
        path = os.path.join(data_dir, file)
        img = plt.imread(path)
        imgs.append(img)
        ids.append(int(file.split(".")[0]))

imgs = np.array(imgs)
ids = np.array(ids)


np.savez('dr16.npz', imgs)
np.savez('dr16_ids.npz', ids)