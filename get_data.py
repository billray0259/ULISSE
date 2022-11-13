
import time

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# Downloads 'size' images, starting from 'offset'
offset = 0
size = 64

# Load in the coordinates
catalog = pd.read_csv('object_info.csv')	
catalog = catalog[offset:offset + size]

ra = catalog['RA'].values
dec = catalog['DEC'].values

imageList = []
coords = []
err = 0

# Loop over all objects and download the thumbnail. If it fails (e.g. internet failure), it retries 5 times, otherwise moves on.
for idx in tqdm(range(len(ra))):
    url = 'http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=' + str(ra[idx]) + '&dec=' + str(dec[idx]) + '&scale=0.4&height=64&width=64'
    for i in range(5):  # retry loop
        try:
            im = imageio.imread(url)
            break  # On success, stop retry.
        except:
            print('timeout, retry in 1 second.')
            time.sleep(1)
            err = 1

    if err == 0:
        imageList.append(im)
        coords.append(idx + offset)

    err = 0

# Save images and indices (in case some objects fail, we need the index of the object in the original file)
imageList = np.array(imageList)

print(imageList.shape, len(coords))

np.savez('dr16.npz', imageList)
np.savetxt('indices.txt', coords)