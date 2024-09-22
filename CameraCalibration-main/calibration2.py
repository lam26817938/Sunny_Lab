import cv2
import numpy as np
import glob
import pickle

# 準備棋盤格規格
chessboard_size = (9, 6)
# 準備世界坐標系下的物理點
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 存儲棋盤格角點的物理坐標和像素坐標
objpoints = []  # 3D點在世界坐標系中的坐標
imgpointsL = []  # 2D點在左相機圖像平面中的坐標
imgpointsR = []  # 2D點在右相機圖像平面中的坐標

# 加載校正圖像
images_left = glob.glob('CameraCalibration-main/images/LC/*.png')
images_right = glob.glob('CameraCalibration-main/images/RC/*.png')

# 找到棋盤角點並存儲
for img_left, img_right in zip(images_left, images_right):
    imgL = cv2.imread(img_left)
    imgR = cv2.imread(img_right)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # 找到棋盤角點
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)

    if retL and retR:
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

# 單相機校正
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

# 立體校正
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL,
    mtxR, distR,
    grayL.shape[::-1], criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
)

# 立體校正（Stereo Rectification）
print(imgL.shape[::-1])
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, grayL.shape[::-1], R, T)


mapLx, mapLy = cv2.initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1, grayL.shape[::-1], cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2, grayL.shape[::-1], cv2.CV_32FC1)

# 对720p图像进行去畸变和校正
undistortedL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
undistortedR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

# Display the undistorted and rectified images
cv2.imshow('Undistorted and Rectified Left Image (720p)', undistortedL)
cv2.imshow('Undistorted and Rectified Right Image (720p)', undistortedR)

cv2.waitKey(0)

# 保存校正信息到文件
calibration_data = {
    'cameraMatrixL': cameraMatrixL,
    'distCoeffsL': distCoeffsL,
    'cameraMatrixR': cameraMatrixR,
    'distCoeffsR': distCoeffsR,
    'R': R,
    'T': T,
    'R1': R1,
    'R2': R2,
    'P1': P1,
    'P2': P2,
    'Q': Q
}

# # 使用pickle保存
# with open('stereo_calibration_data.pkl', 'wb') as f:
#     pickle.dump(calibration_data, f)

# 使用numpy保存
np.savez('CameraCalibration-main/stereo_calibration_data_720.npz',
         cameraMatrixL=cameraMatrixL,
         distCoeffsL=distCoeffsL,
         cameraMatrixR=cameraMatrixR,
         distCoeffsR=distCoeffsR,
         R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)