import torch
from torchmetrics.detection import MeanAveragePrecision


#################### "2D evaluation with original metric" ####################
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

metric = MeanAveragePrecision(class_metrics=True, extended_summary=False)
metric.update(preds, target)
from pprint import pprint
pprint(metric.compute())

############################################################################

########################### "3D evaluation" ################################









