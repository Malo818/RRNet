import os
import json
import argparse
import pickle
from build_dataset import MultiViewPointCloudDataset


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="../config.json", help="path to the json config file")
args = parser.parse_args()
config = args.config
args = json.load(open(config))
device = args["device"]

file_list = os.listdir(args["data"])

imagess = []
pointss = []
labelss = []

i =1
for name in file_list:
    file_path = os.path.join(args["data"], name)
    print("Number " + str(i) + " scene is being generated")
    dataset = MultiViewPointCloudDataset(file_path, args["d_point"],args["imagesize"],args["point_density"])
    imagess.append(dataset.images)
    pointss.append(dataset.points)
    labelss.append(dataset.labels)
    i +=1

with open('scenenn.pkl', 'wb') as f:
    pickle.dump({'image': imagess,'point': pointss, 'label': labelss}, f)

