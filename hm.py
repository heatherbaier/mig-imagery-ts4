import os

im_dir = os.path.join(os.getcwd(), "MEX2imagery", "2010")

for month in os.listdir(im_dir):
    print(month, len(os.listdir(os.path.join(im_dir, month))))