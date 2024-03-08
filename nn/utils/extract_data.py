# script to extract data from h5py files and save it in a format that can be used for training (npz files)

import h5py
import os
from torch.utils.data import Dataset
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import argparse


#argparse
parser = argparse.ArgumentParser(description='Extract data from h5py files')
parser.add_argument('--dataset_path', type=str, help='Path to the dataset',
                    default='/home/dennis/git_repos/multiview_detection_multisense/data/converted_object_detection_dataset_imu')
parser.add_argument('--target_path', type=str, help='Path to the target folder',
                    default='/home/dennis/git_repos/mslp-dataset/benchmark/nn/')


def main():
    args = parser.parse_args()
    dataset_path = args.dataset_path
    target_path = args.target_path

    target_path = os.path.join(target_path, "dataset")
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")

    target_train_path = os.path.join(target_path, "train")
    target_test_path = os.path.join(target_path, "test")
    os.makedirs(target_train_path, exist_ok=True)
    os.makedirs(target_test_path, exist_ok=True)

    classes = {"motorboat": 1, "sailboat": 2, "sailboat under bare poles": 3, "stand-up-paddle": 4, "catamaran": 5,
                        "ferry": 6, "pedal boat": 7, "motor vessel": 8, "pile": 9}

    for c, path in enumerate([train_path, test_path]):
        for file in tqdm(sorted(os.listdir(path))):
            with h5py.File(os.path.join(path, file), "r") as f:

                img_l = f['left_image/image'][:] / np.iinfo("uint16").max

                K_l = f['left_image/K'][:].reshape(3,3)
                D_l = f['left_image/D'][:]
                R_l = f['left_image/R'][:].reshape(3,3)
                P_l = f['left_image/P'][:].reshape(3,4)

                img_l_undistorted = cv2.undistort(img_l, K_l, D_l)

                # map from plane to velodyne
                R_plane_cam = f["calib_cam_to_plane/R"][:]
                t_plane_cam = f["calib_cam_to_plane/t"][:]
                Tr_cam_plane = np.zeros((3,4))
                Tr_cam_plane[:3,:3] = R_plane_cam.T
                Tr_cam_plane[:,3] = -(R_plane_cam.T @ t_plane_cam)

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
                        label_int = classes[label_str]
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

                
                bounding_boxes = []
                for i in range(len(bbox_location)):
                    w, h, l = bbox_dimensions[i][0], bbox_dimensions[i][1], bbox_dimensions[i][2]
                    location = bbox_location[i]
                    rotation_y = bbox_rotation_y[i]
                    bbox_points = np.array([[-w/2, h/2, l/2], [w/2, h/2, l/2], [w/2, h/2, -l/2], [-w/2, h/2, -l/2],
                                            [-w/2, -h/2, l/2], [w/2, -h/2, l/2], [w/2, -h/2, -l/2], [-w/2, -h/2, -l/2]])
                    rotation_matrix = R.from_euler("y", rotation_y, degrees=False).as_matrix()
                    bbox_points = (rotation_matrix @ np.expand_dims(bbox_points, axis=2)).squeeze(2)
                    bbox_points += location
                    
                    ones = np.ones((bbox_points.shape[0], 1))
                    bbox_points_hom = np.concatenate((bbox_points, ones), axis=1)

                    # transform to camera
                    cam_coords_corners3d = np.matmul(Tr_cam_plane, np.expand_dims(bbox_points_hom, axis=2)).squeeze(axis=2)

                    # # go back from camera rect to camera coordinate system
                    cam_coords_corners = np.matmul(R_l.T, np.expand_dims(cam_coords_corners3d, axis=2)).squeeze(axis=2)
                    # # go into image
                    img_coords_corners = np.matmul(K_l, np.expand_dims(cam_coords_corners, axis=2)).squeeze(axis=2)
                    img_coords_corners = np.true_divide(img_coords_corners[:,:2], img_coords_corners[:,[-1]])

                    lower_bound = lambda x : 0 if (x <= 0) else x
                    upper_x_bound = lambda x : img_l.shape[1]-1 if (x >= img_l.shape[1]) else x
                    upper_y_bound = lambda x : img_l.shape[0]-1 if (x >= img_l.shape[0]) else x
                    
                    left = lower_bound(img_coords_corners[:,0].min())
                    right = upper_x_bound(img_coords_corners[:,0].max())
                    top = lower_bound(img_coords_corners[:,1].min())
                    bottom = upper_y_bound(img_coords_corners[:,1].max())

                    bounding_boxes.append([int(left), int(top), int(right), int(bottom)])
                
                # save image
                if c == 0:
                    save_path = target_train_path
                else:
                    save_path = target_test_path
                save_file_name = file.split(".")[0]
                np.savez(os.path.join(save_path, save_file_name), img=(img_l_undistorted*255).astype("uint8"), bbox=bounding_boxes, label=bbox_label_int)


if __name__ == "__main__":
    main()