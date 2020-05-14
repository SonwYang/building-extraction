import json
from visdom import Visdom

def vis_loss(filepath):
    f = open(filepath)
    viz = Visdom()
    d = json.load(f)
    viz.line([[0.0, 0.0]], [0.], win="loss", opts=dict(title="train_valid_loss", legend=['trainLoss', 'validLoss']))
    for i in range(len(d["train loss"])):
        viz.line([[d["train loss"][i], d["val loss"][i]]], [i+1], win='loss', update='append')

def vis_miou(filepath):
    f = open(filepath)
    viz = Visdom()
    d = json.load(f)
    viz.line([[0.0, 0.0]], [0.], win="miou", opts=dict(title="train_valid_miou", legend=['trainMiou', 'validMiou']))
    for i in range(len(d["train miou"])):
        viz.line([[d["train miou"][i], d["val miou"][i]]], [i+1], win='miou', update='append')

def vis_miou2(filepath1,filepath2):
    f1 = open(filepath1)
    f2 = open(filepath2)
    viz = Visdom()
    d1 = json.load(f1)
    d2 = json.load(f2)
    viz.line([[0.0, 0.0, 0.0, 0.0]], [0.], win="miou", opts=dict(title="train_valid_miou", legend=['trainMiou',  'validMiou', "trainMiou_wm", "val miou_wm",]))
    for i in range(len(d1["train miou"])):
        viz.line([[d1["train miou"][i], d1["val miou"][i], d2["train miou"][i], d2["val miou"][i]]], [i+1], win='miou', update='append')

path1 = "urban3d_4c.json"
path2 = "urban3d_4c_wm.json"
vis_miou2(path1,path2)
