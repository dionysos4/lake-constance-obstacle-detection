import torch
import torchvision
from pytorch_lightning.core import LightningModule
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class LitRetinaNet(LightningModule):
    def __init__(self, num_classes, min_size, max_size,
                 lr=0.001, weight_decay=0.0001):
        super().__init__()
        self.model = torchvision.models.detection.retinanet_resnet50_fpn(weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT, min_size=min_size, max_size=max_size,
                                                                          num_classes=91, trainable_backbone_layers=5)
        self.save_hyperparameters()
        
        # set input transform
        self.model.transform.min_size = (min_size,)
        self.model.transform.max_size = max_size
        
        # replace the pre-trained classification head with a new one
        num_anchors = self.model.head.regression_head.bbox_reg.out_channels // 4
        in_features = self.model.head.classification_head.cls_logits.in_channels
        self.model.head.classification_head.cls_logits = torch.nn.Conv2d(in_features, num_anchors*num_classes, stride=1, kernel_size=3, padding=1)
        self.model.num_classes = num_classes
        self.model.head.classification_head.num_classes = num_classes

        self.lr = lr
        self.weight_decay = weight_decay


    def forward(self, x, targets=None):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        output = self.model(x, targets)
        return output
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=6,
                                                                     factor=.1, cooldown=0, min_lr=1e-7)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "validation/loss",
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },
    }

    def training_step(self, batch, batch_idx):
        img, targets = batch["left_img"], batch["targets"]
        converted_targets = self.convert_for_network_input(targets)
        losses = self(img, converted_targets)
        loss = losses["classification"] + losses["bbox_regression"]
            
        self.log("training/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("training/classification_loss", losses["classification"], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("training/regression_loss", losses["bbox_regression"], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # set model and submodule to training for loss output
        self.model.training = True

        img, targets = batch["left_img"], batch["targets"]
        converted_targets = self.convert_for_network_input(targets)
        losses = self(img, converted_targets)
        loss = losses["classification"] + losses["bbox_regression"]
            
        self.log("validation/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("validation/classification_loss", losses["classification"], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("validation/regression_loss", losses["bbox_regression"], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return loss
        
    def convert_for_network_input(self, targets, batch_size=1):
        boxes = targets[0]
        labels = targets[1]
        batch_size = boxes.shape[0]
        converted_targets = []
        for i in range(batch_size):
            mask = boxes[i].sum(axis=1) != 0
            d = {}
            d["boxes"] = boxes[i][mask].long()
            d["labels"] = labels[i][mask].long()
            converted_targets.append(d)
        return converted_targets