import torch
import torchvision
from utils.dataset import MARITIMEDETECTION
from models.faster_rcnn import LitFasterRCNN
from utils.transforms import ToTensor
import matplotlib.pyplot as plt

val_dataset = MARITIMEDETECTION("/media/dennis/3B9FC6C559F7A944/converted_object_detection_dataset", "test",
                                    transform=torchvision.transforms.Compose([ 
                                    ToTensor()]))

model = LitFasterRCNN.load_from_checkpoint("/home/dennis/git_repos/mslp-dataset/benchmark/nn/faster_rcnn_log/version_0/fasterrcnn-epoch=38-val_loss=0.00.ckpt", map_location="cuda", min_size=800, max_size=900)
model.eval()

for sample in val_dataset:
    img, targets = sample["left_img"], sample["targets"]

    output = model(img.cuda(), None)

    scores = output[0]["scores"].cpu()
    boxes = output[0]["boxes"].cpu()
    labels = output[0]["labels"].cpu()

    score_mask = scores > 0.8
    scores = scores[score_mask]
    boxes = boxes[score_mask]
    labels_int = labels[score_mask]
    labels = [str(l) for l in labels_int.tolist()]

    image = sample["left_img"]
    image = (image*255).to(torch.uint8)
    # plot prediction
    img = torchvision.utils.draw_bounding_boxes(image, boxes, labels, colors=(255,0,0))
    # plot ground truth
    img = torchvision.utils.draw_bounding_boxes(img, targets[0], targets[1].numpy().astype("str").tolist(), colors=(0,255,0))

    plt.imshow(img.numpy().transpose((1,2,0)))
    plt.show()