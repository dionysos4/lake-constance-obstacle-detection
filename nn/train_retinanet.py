from utils.dataset import MARITIMEDETECTION
from models.retina_net import LitRetinaNet
from utils.transforms import ToTensor, HorizontalFlip
from torch.utils.data import DataLoader
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.trainer import Trainer
import torch
import matplotlib.pyplot as plt
import os
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    torch.manual_seed(1133)
    LOG_PATH = "/home/dennis/git_repos/mslp-dataset/benchmark/nn"
    LOG_DIR = "retina_net_log"
    VERSION = 0
    text_to_save_with_experiment = ['Readme', 'changed lr and added horizontal flip in training']

    logging_dir = os.path.join(os.path.join(LOG_PATH, LOG_DIR), "version_" + str(VERSION))
    if os.path.exists(logging_dir):
        print("Experiment already conducted!")
        exit(-1)
    os.mkdir(logging_dir)
    with open(os.path.join(logging_dir, 'README.txt'), 'w') as f:
        f.write('\n'.join(text_to_save_with_experiment))

    data_dict = {"left_img" : 1, "right_img" : 1, "point_cloud" : 1,  "calibration" : 1, "annotation": 1, "scene_idx" : 1}


    train_dataset = MARITIMEDETECTION("/home/dennis/object_detection_dataset", "training",    
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.RandomApply([HorizontalFlip()], p=0.5),
                                        ToTensor()]))

    dataloader_train = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)

    val_dataset = MARITIMEDETECTION("/home/dennis/object_detection_dataset", "validation",
                                        transform=torchvision.transforms.Compose([
                                        ToTensor()]))

    dataloader_val = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=16)


    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback_val = ModelCheckpoint(
                            save_top_k=1,
                            monitor="validation/loss",
                            mode="min",
                            dirpath=logging_dir,
                            filename="retinanet-{epoch:02d}-{val_loss:.2f}",
    )

    checkpoint_callback_train = ModelCheckpoint(
                            save_top_k=1,
                            monitor="training/loss",
                            mode="min",
                            dirpath=logging_dir,
                            filename="retinanet-{epoch:02d}-{train_loss:.2f}",
    )

    early_stop_callback = EarlyStopping(monitor="validation/loss", min_delta=0.000, patience=10, verbose=False, mode="min")
    logger = TensorBoardLogger(LOG_PATH, name=LOG_DIR, version=VERSION)
    # 10 classes, class 0 background
    model = LitRetinaNet(num_classes=10, min_size=800, max_size=900, lr=0.0001, weight_decay=0.001)
    trainer = Trainer(callbacks=[checkpoint_callback_val, checkpoint_callback_train, early_stop_callback], logger=logger, num_sanity_val_steps=0, num_nodes=1, strategy="ddp", precision=32, log_every_n_steps=1, accelerator="gpu", devices=[3])
    trainer.fit(model, dataloader_train, dataloader_val)


if __name__ == "__main__":
    main()
