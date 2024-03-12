from utils.faster_dataset import LakeConstance2dDataset
from models.faster_rcnn import LitFasterRCNN
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
import argparse


#argparse
parser = argparse.ArgumentParser(description='train faster rcnn')
parser.add_argument('--log_path', type=str, help='Path where log_information is stored',
                    default='nn')
parser.add_argument('--dataset_path', type=str, help='Path to the npz files',
                    default='nn/dataset')

def main():
    torch.set_float32_matmul_precision('high')
    args = parser.parse_args()
    log_path = args.log_path

    torch.manual_seed(1456)
    LOG_PATH = log_path
    DATA_PATH = args.dataset_path
    LOG_DIR = "faster_rcnn_log_new"
    VERSION = 5
    text_to_save_with_experiment = ['Readme', 'training again']

    logging_dir = os.path.join(os.path.join(LOG_PATH, LOG_DIR), "version_" + str(VERSION))
    if os.path.exists(logging_dir):
        print("Experiment already conducted!")
        exit(-1)
    os.mkdir(logging_dir)
    with open(os.path.join(logging_dir, 'README.txt'), 'w') as f:
        f.write('\n'.join(text_to_save_with_experiment))


    train_dataset = LakeConstance2dDataset(DATA_PATH, "train",
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.RandomApply([HorizontalFlip()], p=0.5),
                                        ToTensor()]))

    dataloader_train = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=6)

    val_dataset = LakeConstance2dDataset(DATA_PATH, "val",
                                        transform=torchvision.transforms.Compose([
                                        ToTensor()]))

    dataloader_val = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=6)


    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback_val = ModelCheckpoint(
                            save_top_k=1,
                            monitor="validation/loss",
                            mode="min",
                            dirpath=logging_dir,
                            filename="fasterrcnn-{epoch:02d}-{val_loss:.2f}",
    )

    checkpoint_callback_train = ModelCheckpoint(
                            save_top_k=1,
                            monitor="training/loss",
                            mode="min",
                            dirpath=logging_dir,
                            filename="fasterrcnn-{epoch:02d}-{train_loss:.2f}",
    )

    early_stop_callback = EarlyStopping(monitor="validation/loss", min_delta=0.000, patience=10, verbose=False, mode="min")
    logger = TensorBoardLogger(LOG_PATH, name=LOG_DIR, version=VERSION)
    # 10 classes, class 0 background
    model = LitFasterRCNN(num_classes=10, min_size=800, max_size=900, lr=0.00005, weight_decay=0.0001)
    trainer = Trainer(callbacks=[checkpoint_callback_val, checkpoint_callback_train, early_stop_callback], logger=logger, num_sanity_val_steps=0, num_nodes=1, strategy="ddp", precision=32, log_every_n_steps=1, accelerator="gpu", devices=[1])
    trainer.fit(model, dataloader_train, dataloader_val)


if __name__ == "__main__":
    main()