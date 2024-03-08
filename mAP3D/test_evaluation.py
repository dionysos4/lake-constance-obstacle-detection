import torch
import numpy as np
from metric.detection import MeanAveragePrecision

preds = [
    dict(
        boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
        scores=torch.tensor([0.536]),
        labels=torch.tensor([0]),
    )
]
target = [
    dict(
        boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
        labels=torch.tensor([0]),
    )
]


preds.append(
    dict(
        boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0], [0, 0, 200, 200]]),
        scores=torch.tensor([0.9, 0.8]),
        labels=torch.tensor([0, 1]),
    )
)

target.append(
    dict(
        boxes=torch.tensor([[258.0, 41.0, 400.0, 200.0], [0, 0, 101, 200]]),
        labels=torch.tensor([0, 1]),
    )
)


preds.append(
    dict(
        boxes=torch.tensor([]),
        scores=torch.tensor([]),
        labels=torch.tensor([]),
    )
)

target.append(
    dict(
        boxes=torch.tensor([[258.0, 41.0, 400.0, 200.0], [0, 0, 101, 200]]),
        labels=torch.tensor([0, 1]),
    )
)

metric = MeanAveragePrecision(class_metrics=True)
metric.update(preds, target)
from pprint import pprint
pprint(metric.compute())

preds_3d = []
# convert 2d boxes to 3d
for detection_per_img in range(len(preds)):
    # if no detection exist
    if len(preds[detection_per_img]["boxes"]) == 0:
        preds_3d.append(dict(
        boxes=torch.tensor([]),
        scores=torch.tensor([]),
        labels=torch.tensor([]),
    ))
    for j, box in enumerate(preds[detection_per_img]["boxes"]):
        xmin=box[0]
        ymin=box[1]
        xmax=box[2]
        ymax=box[3]

        h = 4
        new_box = torch.tensor([[xmin, ymin, 0], [xmax, ymin, 0], [xmax, ymax, 0], [xmin, ymax, 0], [xmin, ymin, h], [xmax, ymin, h], [xmax, ymax, h], [xmin, ymax, h]])
        if j == 0:
            preds_3d.append(dict(
                boxes=new_box.unsqueeze(dim=0),
                scores=preds[detection_per_img]["scores"],
                labels=preds[detection_per_img]["labels"],
            ))
        else:
            preds_3d[detection_per_img]["boxes"] = torch.stack((preds_3d[detection_per_img]["boxes"], new_box.unsqueeze(0)), dim=1).squeeze()


target_boxes_3d = []
for targets_per_img in range(len(target)):
    # if no target exist
    if len(target[targets_per_img]["boxes"]) == 0:
        target_boxes_3d.append(dict(
        boxes=torch.tensor([]),
        scores=torch.tensor([]),
        labels=torch.tensor([]),
    ))
    for j, box in enumerate(target[targets_per_img]["boxes"]):
        xmin=box[0]
        ymin=box[1]
        xmax=box[2]
        ymax=box[3]

        h = 4
        new_box = torch.tensor([[xmin, ymin, 0], [xmax, ymin, 0], [xmax, ymax, 0], [xmin, ymax, 0], [xmin, ymin, h], [xmax, ymin, h], [xmax, ymax, h], [xmin, ymax, h]])
        
        if j == 0:
            target_boxes_3d.append(dict(
                boxes=new_box.unsqueeze(dim=0),
                labels=target[targets_per_img]["labels"],
            ))
        else:
            target_boxes_3d[targets_per_img]["boxes"] = torch.stack((target_boxes_3d[targets_per_img]["boxes"], new_box.unsqueeze(0)), dim=1).squeeze()


metric = MeanAveragePrecision(class_metrics=True)
metric.update(preds_3d, target_boxes_3d)
from pprint import pprint
pprint(metric.compute())