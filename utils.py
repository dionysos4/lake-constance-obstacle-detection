import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor, LinearRegression


def create_T_matrix(R, t):
    """ Create homogenous 4x4 transformation matrix

    Parameters
    ----------
    R : np.array
        3 x 3 rotation matrix
    t : np.array
        1 x 3 translation vector
    
    return: np.array
        4 x 4 transformation matrix
    """ 
    T = np.zeros((4,4))
    T[3,3] = 1
    T[:3,:3] = R
    T[:3, 3] = t
    return T


def rectify_img(left_img, right_img, K_l, D_l, R_l, P_l, K_r, D_r, R_r, P_r):
    """ Rectifies a stereo image pair

    Parameters
    ----------
    left_img : np.array
        left raw image
    right_img : np.array
        right raw
    K_l : np.array
        intrinsics left camera
    D_l : np.array
        distortion coefficients left camera
    R_l : np.array
        rectification matrix of the left camera
    P_l : np.array
        projection matrix of the left camera
    K_r : np.array
        intrinsics right camera
    D_r : np.array
        distortion coefficients right camera
    R_r : np.array
        rectification matrix of the right camera
    P_r : np.array
        projection matrix of the right camera
    
    return: tuple
        rectified images of left and right camera
    """ 
    left_img_float = left_img / np.iinfo("uint16").max
    right_img_float = right_img / np.iinfo("uint16").max

    map1_left, map2_left = cv2.initUndistortRectifyMap(K_l, D_l, R_l, P_l, (left_img.shape[1], left_img.shape[0]), cv2.CV_32FC1)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K_r, D_r, R_r, P_r, (right_img.shape[1], right_img.shape[0]), cv2.CV_32FC1)

    left_img_rect = cv2.remap(left_img_float, map1_left, map2_left, cv2.INTER_LINEAR)
    right_img_rect = cv2.remap(right_img_float, map1_right, map2_right, cv2.INTER_LINEAR)
    return left_img_rect, right_img_rect



def compute_pointcloud(left_img, right_img, P_left, P_right, minDisparity = 0, numDisparities=512, blocksize=17):
    """ Generates pointcloud from stereo images by usage of SGBM

    Parameters
    ----------
    left_img : np.array
        left rectified image
    right_img : np.array
        right rectified image
    P_left : np.array
        projection matrix of the left camera
    P_right : np.array
        projection matrix of the right camera
    minDisparity : int
        minimum disparity
    numDisparities : int
        number of disparities
    block_size : int
        block size
    
    return: tuple
        point cloud and disparity map
    """ 
    gray_l = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
    gray_r = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
    gray_l = gray_l.astype("uint8")
    gray_r = gray_r.astype("uint8")
    stereo = cv2.StereoSGBM_create(minDisparity=minDisparity, numDisparities=numDisparities, blockSize=blocksize, preFilterCap=31, uniquenessRatio=15, P1=200, P2=400, speckleRange=4, speckleWindowSize=100)
    disp = stereo.compute(gray_l, gray_r)

    c_x = P_left[0,2]
    c_y = P_left[1,2]
    f = P_left[0,0]
    Tx = (P_right[:,3] / P_right[0,0])[0]
    c_x_ = P_right[0,2]
    Q = np.array([[1, 0, 0, -c_x], [0, 1, 0, -c_y], [0, 0, 0, f], [0, 0, -1/Tx, (c_x - c_x_)/Tx]])

    disp = disp / 16.0
    disp = disp.astype("float32")
    pc_img = cv2.reprojectImageTo3D(disp, Q)

    inf_mask_x = np.isfinite(pc_img[:,:,0])
    inf_mask_y = np.isfinite(pc_img[:,:,1])
    inf_mask_z = np.isfinite(pc_img[:,:,2])
    z_mask = pc_img[:,:,2] > 0

    inf_mask = inf_mask_x & inf_mask_y & inf_mask_z & z_mask
    pc = pc_img[inf_mask]
    rgb = left_img[inf_mask]
    pc = np.concatenate((pc, rgb), axis=1)
    pc = pc
    return pc, disp


def calc_plane(point_cloud, var_index=[0,1,2]):
    '''
    Calculates a regression plane in given point cloud using RANSAC with LinearRegression.
    var_index determines which columns contain the dependant and the independant variables.
    Last entry of the array is the index of the dependant variable.
    The axis of the dependant variable must not be parallel to the regression plane. 
    Parameters
    ----------
    point_cloud : np.array
        left rectified image
    var_index : list
        axis selection
    
    return: tuple
        plane normal, support point
    '''
    y = point_cloud[:,var_index[2]]
    X = -point_cloud[:,var_index[0:2]]
    lin_reg = LinearRegression()
    reg = RANSACRegressor(estimator=lin_reg, stop_probability=0.99, residual_threshold=0.2,max_trials=1000).fit(X, y)
    normal_vec, support_vec = ([0,0,0], [0,0,0])
    normal_vec[var_index[2]] = 1
    normal_vec[var_index[0]] = reg.estimator_.coef_[0]
    normal_vec[var_index[1]] = reg.estimator_.coef_[1]
    support_vec[var_index[2]] = reg.estimator_.intercept_

    return normal_vec, support_vec


# converts bounding box from open3d format to pytorch 3d format
def o3d_to_pyt3d(p):
    conv_dict = {0 : 7, 1 : 4, 2 : 6, 3 : 1, 4 : 2, 5 : 5, 6 : 3, 7 : 0}
    idx = list(conv_dict.values())
    return p[idx]