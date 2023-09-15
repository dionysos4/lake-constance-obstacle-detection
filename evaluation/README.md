### Fork from torchmetrics to compute mean average precision with 3d bounding boxes

How to use the module
```
from metric.detection import MeanAveragePrecision

metric = MeanAveragePrecision(class_metrics=True)
preds = [
    dict(
        boxes=torch.FloatTensor([[[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1]]]),
        scores=torch.FloatTensor([0.536]),
        labels=torch.tensor([0]),
    )
]
target = [
    dict(
        boxes=torch.FloatTensor([[[-1.5, 1, 0], [1, 1, 0], [1, -1, 0], [-1.5, -1, 0], [-1.5, 1, 1], [1, 1, 1], [1, -1, 1], [-1.5, -1, 1]]]),
        labels=torch.tensor([0]),
    )
]
metric.update(preds, target)
from pprint import pprint
pprint(metric.compute())
```

Input to the metric are the vertices of the boxes in the following order

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)