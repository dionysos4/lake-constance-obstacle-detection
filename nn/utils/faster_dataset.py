import numpy as np
from torch.utils.data import Dataset
import os


class LakeConstance2dDataset(Dataset):
    """
    Pytorch LakeConstace2dDataset class to load the dataset

    Parameters
    ----------
    dataset_dir : str
        directory where the dataset is stored
    training : string
        choose train, val or test dataset
    transform : torch transform objects
        transformation which are applied to the input
    
    return: dict
        {'left_img'; 'targets'}
    """

    def __init__(self, dataset_dir, training, transform):
        self.dataset_dir = dataset_dir

        self.transform = transform
        self.training = training

        self.test_dir = os.path.join(self.dataset_dir, "test")
        self.test_files = sorted(os.listdir(self.test_dir))

        self.train_dir = os.path.join(self.dataset_dir, "train")
        tmp_train_files = sorted(os.listdir(self.train_dir))
        np.random.seed(200)
        np.random.shuffle(tmp_train_files)

        self.idx_boarder_train = int(len(tmp_train_files) / 100 * 80)

        self.train_files = tmp_train_files[:self.idx_boarder_train]
        self.valid_files = tmp_train_files[self.idx_boarder_train:]

    #     #mean_l, std_l, mean_r, std_r = self.__compute_normalization()


    def __len__(self):
        if self.training == "train":
            return len(self.train_files)
        elif self.training == "val":
            return len(self.valid_files)
        else:
            return len(self.test_files)


    def __getitem__(self, idx):
        if self.training == "train":
            data = os.path.join(self.train_dir, self.train_files[idx])
        elif self.training == "val":
            data = os.path.join(self.train_dir, self.valid_files[idx])
        else:
            data = os.path.join(self.test_dir, self.test_files[idx])

        data = np.load(data)
        img = data["img"]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        bbox = data["bbox"]
        label = data["label"]

        targets = {}
        targets["boxes"] = np.asarray(bbox)
        targets["labels"] = np.asarray(label)

        sample = {'left_img' : img, 'targets' : targets}
        if self.transform:
            sample = self.transform(sample)
        return sample