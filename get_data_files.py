import pandas as pd
import os
# use threadpool to download images
from multiprocessing.pool import ThreadPool
# import imageio.v2 as imageio
import requests
log_file = "errors.log"


def download_image(ra, dec):
    save_file = f"galaxy_files/{ra}_{dec}.jpg"
    if os.path.exists(save_file):
        return
    
    url = f'http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale=0.4&height=64&width=64'
    
    try:
        response = requests.get(url, timeout=30)
        with open(save_file, 'wb') as f:
            f.write(response.content)

        print(f"Saved {save_file} {len(os.listdir('galaxy_files'))}")
    except:
        print(f'Error saving image for {ra}, {dec}')
        with open(log_file, 'a') as f:
            f.write(f'{ra},{dec}')


catalog = pd.read_csv('object_info.csv')

# seed pandas .sample() function and sample 100k rows
catalog = catalog.sample(frac=1, random_state=42).reset_index(drop=True)
catalog = catalog[:100000]

ra = catalog['ra'].values
dec = catalog['dec'].values

# download images
pool = ThreadPool(10)
pool.starmap(download_image, zip(ra, dec))
pool.close()
pool.join()



