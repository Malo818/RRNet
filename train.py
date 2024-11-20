import os
import json
import datetime
import argparse
import numpy as np
import torch, gc
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from collections import defaultdict
from data.dataset import localdataset
from utils.datapro import data2tensor
from utils.draw import draw_point, draw_pic
from utils.merge_sort import lobalclassification, pointsclassification
from models.network import RRNet
from models.losses import *

gc.collect()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.json", help="path to the json config file")
parser.add_argument("--logdir", default="logs/RR", help="path to the log directory")
args = parser.parse_args()
config = args.config
logdir = args.logdir
args = json.load(open(config))
device = args["device"]

if not os.path.exists(logdir):
    os.mkdir(logdir)
fname = os.path.join(logdir, "config.json")
with open(fname, "w") as fp:
    json.dump(args, fp, indent=4)
datasets = localdataset("data/scenenn.pkl")

model = RRNet(args["embedding_size"])
model.to(device)
lcl = lobalclassification()
lcl.to(device)
pcl = pointsclassification()
pcl.to(device)

criterion = {
    "triplet": HardTripletLoss(args["margin"], args["hardest"]),
    "bce": BCELoss(args["batch_size"]),
    "mse": MSELoss(),
}
criterion["triplet"].to(device)
criterion["bce"].to(device)
criterion["mse"].to(device)

all_parameters = list(model.parameters()) + list(lcl.parameters()) + list(pcl.parameters())
optimizer = optim.SGD(
    all_parameters,
    lr=args["learning_rate"],
    momentum=args["momentum"],
    weight_decay=args["weight_decay"],
)

best_loss = np.Inf

if __name__ == '__main__':
    for epoch in range(args["epochs"]):
        start = datetime.datetime.now()
        scalars = defaultdict(list)
        pc_globalcharacteristics = []
        i = 1
        for dataset in datasets:
            loader = data.DataLoader(
                dataset,
                batch_size=args["batch_size"],
                num_workers=args["num_workers"],
                pin_memory=True,
                shuffle=True,
            )
            j = 1
            for images, points, labels in loader:

                t_images = data2tensor(images).to(device)
                t_points = data2tensor(points).to(device)
                t_labels = data2tensor(labels).to(device)

                #draw_pic(t_images[0:4])
                #draw_point(t_points[0], t_labels[0])

                fgi, fgi4, fgp, fp = model(t_images, t_points)                    # t_images（bs×4，64，64，3）       t_points（bs×m，N，3）   t_points（bs×m，N）
                pc_globalcharacteristics.append(fgp)

                """
                ——Sample training when memory is insufficient——
                num_points = fp.size(1)
                num_samples = num_points // 3
                indices = torch.randperm(num_points)[:num_samples]
                fp_sampled = fp[:, indices, :]
                t_labels_sampled = t_labels[:, indices]
                """

                global_score = lcl(fgi, fgp, pc_globalcharacteristics)
                points_score = pcl(fgi, fp)

                loss_s = 0
                loss_g = 0
                loss_p = 0

                loss_s += args["alpha"] * criterion["triplet"](fgi, fgi4)
                loss_g += args["beta"] * criterion["bce"](global_score, len(pc_globalcharacteristics))
                loss_p += args["gamma"] * criterion["mse"](points_score, t_labels)
                loss = loss_s + loss_g + loss_p

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                scalars["loss"].append(loss)
                scalars["loss_s"].append(loss_s)
                scalars["loss_g"].append(loss_g)
                scalars["loss_p"].append(loss_p)

                now = datetime.datetime.now()
                log = "{} | Scene [{:04d}/{:04d}] | Batch [{:04d}/{:04d}] | loss: {:.4f} |"
                log = log.format(now.strftime("%c"), i, len(datasets), j, len(loader), loss.item())
                j +=1
                print(log)
            i += 1

        # Summary after each epoch
        summary = {}
        now = datetime.datetime.now()
        duration = (now - start).total_seconds()
        log = "> {} | Epoch [{:04d}/{:04d}] | duration: {:.1f}s |"
        log = log.format(now.strftime("%c"), epoch, args["epochs"], duration)
        for m, v in scalars.items():
            summary[m] = torch.stack(v).mean()
            log += " {}: {:.4f} |".format(m, summary[m].item())

        fname = os.path.join(logdir, "checkpoint_{:04d}.pth".format(epoch))
        print("> Saving model to {}...".format(fname))
        model_all = {"RR": model.state_dict(), "lcl": lcl.state_dict(), "pcl": pcl.state_dict()}
        torch.save(model_all, fname)

        if summary["loss"] < best_loss:
            best_loss = summary["loss"]
            fname = os.path.join(logdir, "model.pth")
            print("> Saving model to {}...".format(fname))
            model_all = {"RR": model.state_dict(), "lcl": lcl.state_dict(), "pcl": pcl.state_dict()}
            torch.save(model_all, fname)
        log += " best: {:.4f} |".format(best_loss)

        fname = os.path.join(logdir, "train.log")
        with open(fname, "a") as fp:
            fp.write(log + "\n")

        print(log)
        print("--------------------------------------------------------------------------")
