import torch
import numpy as np


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        left_img = sample['left_img']
        targets = sample['targets']

        left_img = torch.from_numpy(left_img.transpose((2, 0, 1)).copy()).float()

        boxes = torch.zeros((10, 4))
        labels = torch.zeros(10)
        
        n_labels = targets["boxes"].shape[0]
        boxes[:n_labels] = torch.from_numpy(targets["boxes"])
        labels[:n_labels] = torch.from_numpy(targets["labels"])
        
        return {'left_img' : left_img,
                'targets' : (boxes, labels)}
    

class HorizontalFlip(object):
    """flips images and labels horizontal"""
    def __call__(self, sample):
        left_img = sample['left_img']
        targets = sample['targets']
        left_img = left_img[:,::-1]
        width = left_img.shape[1]
        boxes = targets["boxes"]
        labels = targets["labels"]

        tmp = width - boxes[:,2]
        boxes[:,2] = width - boxes[:,0]
        boxes[:,0] = tmp

        targets["boxes"] = boxes
        targets["labels"] = labels

        return {'left_img' : left_img,
                'targets' : targets}