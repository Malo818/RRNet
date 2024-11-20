import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from data.dataset import localdataset
from utils.datapro import data2tensor
from utils.draw import draw_point, draw_pic
from utils.merge_sort import lobalclassification, pointsclassification
from models.network import RRNet

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.json", help="path to the json config file")
parser.add_argument("--logdir", default="logs/RR", help="path to the log directory")
args = parser.parse_args()

config = args.config
logdir = args.logdir
args = json.load(open(config))
device = args["device"]

fname = os.path.join(logdir, "model.pth")
print("> Loading model from {}".format(fname))
model_all = torch.load(fname)

model = RRNet(args["embedding_size"])
model.load_state_dict(model_all["RR"])
model.to(device)
model.eval()

lcl = lobalclassification()
lcl.load_state_dict(model_all["lcl"])
lcl.to(device)
lcl.eval()

pcl = pointsclassification()
pcl.load_state_dict(model_all["pcl"])
pcl.to(device)
lcl.eval()

datasets = localdataset("data/test.pkl")
testdatas = datasets[0]         # Select a scene

subset = Subset(testdatas, [0])  # Select a sample

loader = DataLoader(
    subset,
    batch_size=args["batch_size"],
    num_workers=args["num_workers"],
    pin_memory=True,
    shuffle=False,
)

def index_calculation(A, B):
    total_positives_A = torch.sum(A == 1).item()
    true_positives = torch.sum((A == 1) & (B == 1)).item()
    total_positives_B = torch.sum(B == 1).item()
    if total_positives_A > 0:
        precision = true_positives / total_positives_A
    else:
        precision = 0.0
    if total_positives_B > 0:
        recall = true_positives / total_positives_B
    else:
        recall = 0.0
    F1 = 2 * precision * recall / (precision + recall)
    print("P:", precision)
    print("R:", recall)
    print("F1:", F1)


with torch.no_grad():
    for images, points, labels in loader:
        t_images = data2tensor(images).to(device)
        t_points = data2tensor(points).to(device)
        t_labels = data2tensor(labels).to(device)
        fgi, fgi4, fgp, fp = model(t_images, t_points)
        pc_globalcharacteristics = []

        global_score = lcl(fgi, fgp, pc_globalcharacteristics)
        points_score = pcl(fgi, fp)

        threshold = 0.2
        predicted_value = torch.where(points_score > threshold, torch.tensor(1).to(device), torch.tensor(0).to(device))
        predicted_value = predicted_value.int()

        draw_point(t_points[0], t_labels[0])
        draw_point(t_points[0], predicted_value[0][0])
        index_calculation(predicted_value[0][0], t_labels[0])





