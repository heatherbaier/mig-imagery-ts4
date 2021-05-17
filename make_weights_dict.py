import json
import os

BASE_DIR = "MEXimagery/2010/"

weights_dict = {}

for month in os.listdir(os.path.join(os.getcwd(), BASE_DIR)):
    print("Month: ", month)
    for image in os.listdir(os.path.join(os.getcwd(), BASE_DIR, month)):
        b = image.split("_")[0].split("-")[3]
        if b not in weights_dict:
            weights_dict[b] = 1
        else:
            weights_dict[b] += 1
        

print("Total number of images: ", sum(weights_dict.values()))

with open('image_weights.json', 'w') as outfile:
    json.dump(weights_dict, outfile)