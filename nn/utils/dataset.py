from fcntl import DN_RENAME
from gc import DEBUG_LEAK
import h5py
import os
from torch.utils.data import Dataset
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm


class MARITIMEDETECTION(Dataset):
    """
    Pytorch maritime detection dataset

    Parameters
    ----------
    dataset_dir : str
        directory where the dataset is stored
    training : string
        choose training, validation or test dataset (split is 70 / 20 / 10)
    transform : torch transform objects
        transformation which are applied to the input
    
    return: dict
        {'left_img'; 'right_img', 'point_cloud'; 'calibration'; 'annotations'; 'filename'}
    """

    def __init__(self, dataset_dir, training, transform):
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.test_dir = os.path.join(self.dataset_dir, "test")
        self.left_img_list = []
        self.bounding_boxes_list = []
        self.K_l_list = []
        self.D_l_list = []
        self.P_l_list = []
        self.R_l_list = []
        self.Tr_cam_plane = []
        self.transform = transform

        # class 0 for background
        self.classes = {"motorboat": 1, "sailboat": 2, "sailboat under bare poles": 3, "stand-up-paddle": 4, "catamaran": 5,
                    "ferry": 6, "pedal boat": 7, "motor vessel": 8, "pile": 9}

        self.bbox_label_int = []
        self.bbox_label_str = []
        self.bbox_location = []
        self.bbox_dimensions = []
        self.bbox_rotation_y = []
        self.bbox_visibility = []
        self.bbox_occlusion = []
        self.file_list = []

        self.training = training

        # split into training and validation data
        self.train_valid_dataset_size = len(os.listdir(self.train_dir))

        self.idx_boarder_train = int(self.train_valid_dataset_size / 100 * 80)
        ### set seed
        if self.training == "training" or self.training == "validation":
            np.random.seed(10)
            self.idx = np.arange(self.train_valid_dataset_size)
            np.random.shuffle(self.idx)
        else:
            self.idx = np.arange(len(os.listdir(self.test_dir)))

        self.train_valid_filename_list = sorted(os.listdir(self.train_dir))
        self.test_filename_list = sorted(os.listdir(self.test_dir))

        self.__load_data_to_ram()

        self.transform = transform

    #     #mean_l, std_l, mean_r, std_r = self.__compute_normalization()


    def __len__(self):
        if self.training == "training":
            return self.idx_boarder_train
        elif self.training == "validation":
            return self.train_valid_dataset_size - self.idx_boarder_train
        else:
            return len(os.listdir(self.test_dir))


    def __getitem__(self, idx):

        # training, validation or test
        if self.training == "validation":
            idx = self.idx_boarder_train + idx

        if self.training == "training" or self.training == "validation":
            #file_list = self.train_valid_filename_list
            path = self.train_dir
        else:
            #file_list = self.test_filename_list
            path = self.test_dir
        
        f = h5py.File(os.path.join(path, self.file_list[idx]))
        img_l = f['left_image/image'][:1200,:,:]
        img_l_undistorted = cv2.undistort(img_l / np.iinfo("uint16").max, self.K_l_list[idx], self.D_l_list[idx])

        targets = {}
        targets["boxes"] = np.asarray(self.bounding_boxes_list[idx])
        targets["labels"] = np.asarray(self.bbox_label_int[idx])

        sample = {'left_img' : img_l_undistorted, 'targets' : targets}
        if self.transform:
            sample = self.transform(sample)
        return sample


    def __load_data_to_ram(self):
        """
        stores all the data into ram

        """
        if self.training == "training" or self.training == "validation":
            file_list = self.train_valid_filename_list
            path = self.train_dir
        else:
            file_list = self.test_filename_list
            path = self.test_dir
        for k in tqdm(range(len(file_list))):
            f = h5py.File(os.path.join(path, file_list[self.idx[k]]))
            if k == 0:
                img_l = f['left_image/image'][:1199,:,:]

            self.file_list.append(file_list[self.idx[k]])
            self.K_l_list.append(f['left_image/K'][:].reshape(3,3))
            self.D_l_list.append(f['left_image/D'][:])
            self.R_l_list.append(f['left_image/R'][:].reshape(3,3))
            self.P_l_list.append(f['left_image/P'][:].reshape(3,4))

            # map from plane to velodyne
            R_plane_cam = f["calib_cam_to_plane/R"][:]
            t_plane_cam = f["calib_cam_to_plane/t"][:]
            Tr_cam_plane = np.zeros((3,4))
            Tr_cam_plane[:3,:3] = R_plane_cam.T
            Tr_cam_plane[:,3] = -(R_plane_cam.T @ t_plane_cam)
            self.Tr_cam_plane.append(Tr_cam_plane)

            # load markers
            bbox_label_int = []
            bbox_label_str = []
            bbox_location = []
            bbox_dimensions = []
            bbox_rotation_y = []
            bbox_visibility = []
            bbox_occlusion = []

            i = 0
            while True:
                try:
                    label_str = f["bounding_boxes/bounding_box_" + str(i) + "/category/name"][()].decode("utf-8")
                    label_int = self.classes[label_str]
                    bbox_label_int.append(label_int)
                    bbox_label_str.append(label_str)
                    bbox_location.append(f["bounding_boxes/bounding_box_" + str(i) + "/location"][:])
                    bbox_dimensions.append(f["bounding_boxes/bounding_box_" + str(i) + "/dimensions"][:])
                    bbox_rotation_y.append(f["bounding_boxes/bounding_box_" + str(i) + "/rotation_y"][()])
                    bbox_visibility.append(f["bounding_boxes/bounding_box_" + str(i) + "/visibility"][()])
                    bbox_occlusion.append(f["bounding_boxes/bounding_box_" + str(i) + "/occlusion"][()])
                    i += 1
                except:
                    break
            self.bbox_label_int.append(bbox_label_int)
            self.bbox_label_str.append(bbox_label_str)
            self.bbox_location.append(bbox_location)
            self.bbox_dimensions.append(bbox_dimensions)
            self.bbox_rotation_y.append(bbox_rotation_y)
            self.bbox_visibility.append(bbox_visibility)
            self.bbox_occlusion.append(bbox_occlusion)

            
            bounding_boxes = []
            for i in range(len(self.bbox_location[-1])):
                w, h, l = self.bbox_dimensions[-1][i][0], self.bbox_dimensions[-1][i][1], self.bbox_dimensions[-1][i][2]
                location = self.bbox_location[-1][i]
                rotation_y = self.bbox_rotation_y[-1][i]
                bbox_points = np.array([[-w/2, h/2, l/2], [w/2, h/2, l/2], [w/2, h/2, -l/2], [-w/2, h/2, -l/2],
                                        [-w/2, -h/2, l/2], [w/2, -h/2, l/2], [w/2, -h/2, -l/2], [-w/2, -h/2, -l/2]])
                rotation_matrix = R.from_euler("y", rotation_y, degrees=False).as_matrix()
                bbox_points = (rotation_matrix @ np.expand_dims(bbox_points, axis=2)).squeeze(2)
                bbox_points += location
                
                ones = np.ones((bbox_points.shape[0], 1))
                bbox_points_hom = np.concatenate((bbox_points, ones), axis=1)

                # transform to camera
                cam_coords_corners3d = np.matmul(self.Tr_cam_plane[-1], np.expand_dims(bbox_points_hom, axis=2)).squeeze(axis=2)

                # # go back from camera rect to camera coordinate system
                cam_coords_corners = np.matmul(self.R_l_list[-1].T, np.expand_dims(cam_coords_corners3d, axis=2)).squeeze(axis=2)
                # # go into image
                img_coords_corners = np.matmul(self.K_l_list[-1], np.expand_dims(cam_coords_corners, axis=2)).squeeze(axis=2)
                img_coords_corners = np.true_divide(img_coords_corners[:,:2], img_coords_corners[:,[-1]])

                lower_bound = lambda x : 0 if (x <= 0) else x
                upper_x_bound = lambda x : img_l.shape[1]-1 if (x >= img_l.shape[1]) else x
                upper_y_bound = lambda x : img_l.shape[0]-1 if (x >= img_l.shape[0]) else x
                
                left = lower_bound(img_coords_corners[:,0].min())
                right = upper_x_bound(img_coords_corners[:,0].max())
                top = lower_bound(img_coords_corners[:,1].min())
                bottom = upper_y_bound(img_coords_corners[:,1].max())

                bounding_boxes.append([int(left), int(top), int(right), int(bottom)])
            self.bounding_boxes_list.append(bounding_boxes)


# import time
# d = MARITIMEDETECTION("/media/dennis/3B9FC6C559F7A944/converted_object_detection_dataset", "test", False)
# for i in tqdm(range(0,1000, 1)):
#     img = d[i]["left_img"]
#     for k in range(d[i]["targets"]["boxes"].shape[0]):
#         bbox = d[i]["targets"]["boxes"][k]
#         img = cv2.rectangle(img, tuple(bbox[:2].astype("int")), tuple(bbox[2:].astype("int")),(255,0,0),2)
#     plt.imshow(img)
#     plt.show()