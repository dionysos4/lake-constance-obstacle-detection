import os
import h5py
from tqdm import tqdm
import utils
import numpy as np
import cv2
from nn.utils.transforms import ToTensor
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import torch


class StereoEvaluation():
    """
    A class for image lidar evaluation.

    ...

    Attributes
    ----------
    dataset_path : str
        path where dataset is stored
    model : pytorch-lighning model
        neural network
    metric : torchmetric MAP3D
        metric for evaluation
    prediction_threshold : float
        threshold to hold 2d predictions
    """
    def __init__(self, dataset_path, model, metric, prediction_threshold):
        self.dataset_path = dataset_path
        self.model = model
        self.metric = metric
        self.prediction_threshold = prediction_threshold

        self.left_img = []
        self.K_l = []
        self.D_l = []
        self.R_l = []
        self.P_l = []
        self.right_img = []
        self.K_r = []
        self.D_r = []
        self.R_r = []
        self.P_r = []

        self.R_cam_lidar = []
        self.t_cam_lidar = []

        self.bbox_label_int = []
        self.bbox_label_str = []
        self.bbox_location = []
        self.bbox_dimensions = []
        self.bbox_rotation_y = []
        self.bbox_visibility = []
        self.bbox_occlusion = []


    def __read_file(self, filepath):
        """ Create homogenous 4x4 transformation matrix

        Parameters
        ----------
        filepath : str
            path to hdf5 file
        """
        with h5py.File(filepath, "r") as f:
            self.left_img = f["left_image/image"][:]
            self.K_l = f["left_image/K"][:]
            self.D_l = f["left_image/D"][:]
            self.R_l = f["left_image/R"][:]
            self.P_l = f["left_image/P"][:]
                
            self.right_img = f["right_image/image"][:]
            self.K_r = f["right_image/K"][:]
            self.D_r = f["right_image/D"][:]
            self.R_r = f["right_image/R"][:]
            self.P_r = f["right_image/P"][:]

            self.R_cam_lidar = f["calib_lidar_to_cam/R"][:]
            self.t_cam_lidar = f["calib_lidar_to_cam/t"][:]

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
                    bbox_label_int.append(f["bounding_boxes/bounding_box_" + str(i) + "/category/int"][()]+1)
                    bbox_label_str.append(f["bounding_boxes/bounding_box_" + str(i) + "/category/name"][()].decode("utf-8"))
                    bbox_location.append(f["bounding_boxes/bounding_box_" + str(i) + "/location"][:])
                    bbox_dimensions.append(f["bounding_boxes/bounding_box_" + str(i) + "/dimensions"][:])
                    bbox_rotation_y.append(f["bounding_boxes/bounding_box_" + str(i) + "/rotation_y"][()])
                    bbox_visibility.append(f["bounding_boxes/bounding_box_" + str(i) + "/visibility"][()])
                    bbox_occlusion.append(f["bounding_boxes/bounding_box_" + str(i) + "/occlusion"][()])
                    i += 1
                except:
                    break

            self.bbox_label_int = bbox_label_int
            self.bbox_label_str = bbox_label_str
            self.bbox_location = bbox_location
            self.bbox_dimensions = bbox_dimensions
            self.bbox_rotation_y = bbox_rotation_y
            self.bbox_visibility = bbox_visibility
            self.bbox_occlusion = bbox_occlusion


    def __get_T_plane_cam(self, water_pc):
        n, p = utils.calc_plane(water_pc, var_index=[0,2,1])
        n, p = np.array(n), np.array(p)
        n = n / np.linalg.norm(n)

        # compute translation and rotation: equations in paper
        t_plane_cam_estimated = np.array([0, -n @ p, 0])
        R_cam_plane_estimated = np.zeros((3,3))
        z = np.cross(np.array([1,0,0]), n)
        x = np.cross(n, z)

        R_cam_plane_estimated[:, 0] = x
        R_cam_plane_estimated[:, 1] = n
        R_cam_plane_estimated[:, 2] = z

        return utils.create_T_matrix(R_cam_plane_estimated.T, t_plane_cam_estimated)
    

    def __predict_2d_detection(self, img):
        bounding_boxes = []

        sample = ToTensor()({"left_img" : img, "targets" : {'boxes': np.array([[0,0,0,0]]), 'labels': np.array([0])} })
        img = sample["left_img"]

        output = self.model(img.cuda(), None)

        scores = output[0]["scores"].cpu()
        boxes = output[0]["boxes"].cpu()
        labels = output[0]["labels"].cpu()

        score_mask = scores > self.prediction_threshold
        scores = scores[score_mask]
        boxes = boxes[score_mask]
        labels_int = labels[score_mask]
        labels = [str(l) for l in labels_int.tolist()]

        # only for visualization
        # image = sample["left_img"]
        # image = (image*255).to(torch.uint8)
        # img = torchvision.utils.draw_bounding_boxes(image, boxes, labels, colors=(255,0,0))
        # plt.imshow(img.numpy().transpose((1,2,0)))
        # plt.show()            
        bounding_boxes = [bbox.detach().numpy().tolist() for bbox in boxes]
        return bounding_boxes, scores, labels
    

    def __get_point_clusters(self, stereo_pc, bounding_boxes, img_height, img_width, T_cam_velo):
        pc_image = stereo_pc.reshape((img_height, img_width, 6))
        
        point_clusters = []
        for box in bounding_boxes:
            cluster = pc_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            point_clusters.append(cluster)
        return point_clusters
    

    def __get_3d_boxes(self, point_clusters, T_plane_cam):
        detected_boxes = []
        for clustered_pointcloud in point_clusters:
            # make point cluster homogenous and filter points < 250
            clustered_pointcloud = clustered_pointcloud.reshape((-1, 6))
            clustered_pointcloud = clustered_pointcloud[:,:4]
            mask = clustered_pointcloud[:,2] < 250
            clustered_pointcloud = clustered_pointcloud[mask]
            clustered_pointcloud[:,3] = 1
            
            clustered_pointcloud_hom = (T_plane_cam @ np.expand_dims(clustered_pointcloud, axis=2)).squeeze(axis=2)
            clustered_pointcloud = np.true_divide(clustered_pointcloud_hom[:,:3], clustered_pointcloud_hom[:,[-1]])

            if len(clustered_pointcloud) < 4:
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(clustered_pointcloud)
            height = np.min(clustered_pointcloud[:,1])
            #o3d.visualization.draw_geometries([pcd])

            clustered_pointcloud[:,1] = 0.
            clustered_pointcloud[0,1] = -0.000001

            two_d_box = o3d.geometry.OrientedBoundingBox()
            two_d_box = two_d_box.create_from_points(o3d.utility.Vector3dVector(clustered_pointcloud))
            two_d_box.color = np.array([1.,0.,0.])
            
            # imu coordinate different to plane coordinate system
            two_d_box.extent = np.array([two_d_box.extent[0], two_d_box.extent[1], -height])
            two_d_box.center = np.array([two_d_box.center[0], height/2, two_d_box.center[2]])
            detected_boxes.append(two_d_box)
        return detected_boxes
    
    
    def estimate(self):
        for hdf5_file in tqdm(sorted(os.listdir(self.dataset_path))):
            self.__read_file(os.path.join(self.dataset_path, hdf5_file))
            left_img_rect, right_img_rect = utils.rectify_img(self.left_img, 
                                                              self.right_img, 
                                                              self.K_l, self.D_l, self.R_l, self.P_l, 
                                                              self.K_r, self.D_r, self.R_r, self.P_r)
            left_img_rect, right_img_rect = (left_img_rect * 255).astype("uint8"), (right_img_rect * 255).astype("uint8")
            stereo_pc, disp_img = utils.compute_pointcloud(left_img_rect, right_img_rect, 
                                                     self.P_l, self.P_r, 
                                                     minDisparity=5, 
                                                     numDisparities=256, 
                                                     blocksize=5)
            
            # filter point cloud, last 500 rows and points z < 200
            water_pc = stereo_pc[700*1920:]
            water_pc = water_pc[water_pc[:,2] < 200]
            
            # compute T_plane_cam
            T_plane_cam = self.__get_T_plane_cam(water_pc)

            T_cam_velo = utils.create_T_matrix(self.R_cam_lidar, self.t_cam_lidar)

            left_img_undistorted = cv2.undistort(self.left_img / np.iinfo("uint16").max, self.K_l, self.D_l)
            # get neural network prediction for 2d object detection
            bounding_boxes, scores, labels = self.__predict_2d_detection(left_img_undistorted)
            
            ####### plane, point cloud visualization
            # lidar_points_plane_hom = (T_plane_cam @ T_cam_velo @ np.expand_dims(pc_hom, axis=2)).squeeze(2)
            # lidar_points_plane = np.true_divide(lidar_points_plane_hom[:,:3], lidar_points_plane_hom[:,[-1]])

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(lidar_points_plane)
            # viridis = plt.colormaps["jet"]
            # pcd.colors = o3d.utility.Vector3dVector(viridis(self.pc[:,3].astype("int8"))[:,:3])

            # mesh_box = o3d.geometry.TriangleMesh.create_box(width = 500.0, height = 0.001, depth = 500.0)
            # mesh_box = mesh_box.translate(np.array([[-250],[0],[-250]]))

            # o3d_elements = []
            # o3d_elements.append(pcd)
            # o3d_elements.append(mesh_box)
            # o3d.visualization.draw_geometries(o3d_elements)
            ######## bis hierher #######

            
            # find pointclusters
            point_clusters = self.__get_point_clusters(stereo_pc, bounding_boxes, left_img_undistorted.shape[0], left_img_undistorted.shape[1], T_cam_velo)
            # compute bounding box
            detected_boxes = self.__get_3d_boxes(point_clusters, T_plane_cam)

            ##### get detected bbox corners for metric input
            converted_detections = []
            score_list = []
            labels_list = []

            for i, detection in enumerate(detected_boxes):
                detected_bounding_box = np.asarray(detection.get_box_points())
                detected_bounding_box = utils.o3d_to_pyt3d(detected_bounding_box)
                # transform back to camera coordinate system
                detected_bounding_box_hom = np.zeros((8,4))
                detected_bounding_box_hom[:,3] = 1
                detected_bounding_box_hom[:,:3] = detected_bounding_box

                detected_bounding_box_hom = (np.linalg.inv(T_plane_cam) @ np.expand_dims(detected_bounding_box_hom, axis=2)).squeeze(axis=2)
                detected_bounding_box = np.true_divide(detected_bounding_box_hom[:,:3], detected_bounding_box_hom[:,[-1]])
                
                converted_detections.append(detected_bounding_box)
                score_list.append(scores[i].item())
                labels_list.append(int(labels[i]))
            
        
            ##### get ground truth for metric input ####
            gt_bboxes_list = []
            gt_labels = []
            # define bounding boxes in ground plane
            for i in range(len(self.bbox_label_int)):
                w, h, l = self.bbox_dimensions[i][0], self.bbox_dimensions[i][1], self.bbox_dimensions[i][2]
                location = self.bbox_location[i]
                rotation_y = self.bbox_rotation_y[i]
                gt_bbox = np.array([[-w/2, h/2, l/2], [w/2, h/2, l/2], [w/2, h/2, -l/2], [-w/2, h/2, -l/2],
                                        [-w/2, -h/2, l/2], [w/2, -h/2, l/2], [w/2, -h/2, -l/2], [-w/2, -h/2, -l/2]])
                rotation_matrix = Rotation.from_euler("y", rotation_y, degrees=False).as_matrix()
                gt_bbox = (rotation_matrix @ np.expand_dims(gt_bbox, axis=2)).squeeze(2)
                gt_bbox += location
                
                # transform back to camera coordinate system
                gt_bbox_hom = np.zeros((8,4))
                gt_bbox_hom[:,3] = 1
                gt_bbox_hom[:,:3] = gt_bbox

                gt_bbox_hom = (np.linalg.inv(T_plane_cam) @ np.expand_dims(gt_bbox_hom, axis=2)).squeeze(axis=2)
                gt_bbox = np.true_divide(gt_bbox_hom[:,:3], gt_bbox_hom[:,[-1]])
                gt_bboxes_list.append(gt_bbox)
                gt_labels.append(self.bbox_label_int[i])


            ### only for debugging ###
            # if len(converted_detections) > 0:
            #     o3d_elements = []
            #     for gt_box in gt_bboxes_list:
            #         o3d_bbox = o3d.geometry.OrientedBoundingBox()
            #         o3d_bbox = o3d_bbox.create_from_points(o3d.utility.Vector3dVector(gt_box))
            #         o3d_bbox.color = [0,1,0]
            #         o3d_elements.append(o3d_bbox)

            #     for detected_box in converted_detections:
            #         o3d_bbox = o3d.geometry.OrientedBoundingBox()
            #         o3d_bbox = o3d_bbox.create_from_points(o3d.utility.Vector3dVector(detected_box))
            #         o3d_bbox.color = [1,0,0]
            #         o3d_elements.append(o3d_bbox)

            #     pc_hom = np.zeros((self.pc.shape[0], 4))
            #     pc_hom[:,3] = 1
            #     pc_hom[:,:3] = self.pc[:,:3]

            #     lidar_points_cam_hom = (T_cam_velo @ np.expand_dims(pc_hom, axis=2)).squeeze(2)
            #     lidar_points_cam = np.true_divide(lidar_points_cam_hom[:,:3], lidar_points_cam_hom[:,[-1]])

            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(lidar_points_cam)
            #     o3d_elements.append(pcd)

            #     o3d.visualization.draw_geometries(o3d_elements)
            
            
            ### update metric
            preds_3d = []
            # if there is no detection 
            if len(converted_detections) == 0:
                preds_3d.append(dict(
                    boxes=torch.tensor([]),
                    scores=torch.tensor([]),
                    labels=torch.tensor([]),
                ))
                
            for i in range(len(converted_detections)):
                if i == 0:
                    preds_3d.append(dict(
                        boxes=torch.FloatTensor(converted_detections[i]).unsqueeze(dim=0),
                        scores=torch.FloatTensor(score_list),
                        labels=torch.IntTensor(labels_list),
                    ))
                else:
                    preds_3d[0]["boxes"] = torch.vstack((preds_3d[0]["boxes"], torch.FloatTensor(converted_detections[i]).unsqueeze(dim=0))).squeeze()

            target_boxes_3d = []
            # if there are no targets
            if len(gt_bboxes_list) == 0:
                target_boxes_3d.append(dict(
                    boxes=torch.tensor([]),
                    scores=torch.tensor([]),
                    labels=torch.tensor([]),
                ))

            for i in range(len(gt_bboxes_list)):
                if i == 0:
                    target_boxes_3d.append(dict(
                        boxes=torch.FloatTensor(gt_bboxes_list[i]).unsqueeze(dim=0),
                        labels=torch.IntTensor(gt_labels),
                    ))
                else:
                    target_boxes_3d[0]["boxes"] = torch.vstack((target_boxes_3d[0]["boxes"], torch.FloatTensor(gt_bboxes_list[i]).unsqueeze(dim=0))).squeeze()

            self.metric.update(preds_3d, target_boxes_3d)
        return self.metric


# from metric.detection import MeanAveragePrecision
# from nn.models.faster_rcnn import LitFasterRCNN
# metric = MeanAveragePrecision(class_metrics=True)
# model = LitFasterRCNN.load_from_checkpoint("/home/dennis/git_repos/mslp-dataset/benchmark/nn/faster_rcnn_log/version_3/fasterrcnn-epoch=44-val_loss=0.00.ckpt", map_location="cuda")
# model.eval()
# eval = StereoEvaluation("/media/dennis/3B9FC6C559F7A944/converted_object_detection_dataset/test", model, metric, 0.8)
# metric = eval.estimate()
# from pprint import pprint
# pprint(metric.compute())