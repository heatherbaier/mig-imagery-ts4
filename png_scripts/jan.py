import rasterio as rio
from PIL import Image
import numpy as np
import zipfile
import os


def clip_image(im):
    width, height = im.size 
    left, top = 0, 0
    right, bottom = 120, 120
    im1 = im.crop((left, top, right, bottom))     
    return im1


IMAGERY_DIR = "MEXimagery/2010/1"
FULL_DIR = os.path.join(os.getcwd(), IMAGERY_DIR)
TEMP_DIR = os.path.join(os.getcwd(), "jan_temp")
dir_length = len(os.listdir(FULL_DIR))
num = 21191

for zipfolder in os.listdir(FULL_DIR)[21191:]:
    
    try:
    
        print("Image ", str(num), " of ", str(dir_length), "---- Month: January")
        num += 1

        image_name = zipfolder.split(".")[0]

        # Extract the RGB TIFF files into the temporary directory
        with zipfile.ZipFile(os.path.join(FULL_DIR, zipfolder), 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)

        tiff_files = os.listdir(TEMP_DIR)
        b1 = os.path.join(TEMP_DIR, [i for i in tiff_files if i.endswith("B1.tif")][0])
        b2 = os.path.join(TEMP_DIR, [i for i in tiff_files if i.endswith("B2.tif")][0])
        b3 = os.path.join(TEMP_DIR, [i for i in tiff_files if i.endswith("B3.tif")][0])

        b1, b2, b3 = rio.open(b1).read(1), rio.open(b2).read(1), rio.open(b3).read(1)
        lst = [b3, b2, b1]
        stack = np.dstack(lst)
        PIL_image = Image.fromarray(np.uint8(stack)).convert('RGB')
        PIL_image = clip_image(PIL_image)

        PIL_image.save(os.path.join(os.getcwd(), "model_imagery", (image_name + "_JAN" + ".png")))

        [os.remove(os.path.join(TEMP_DIR, i)) for i in os.listdir(TEMP_DIR) if i.startswith("MEX")]
        
    except:
        
        print("Skipping")