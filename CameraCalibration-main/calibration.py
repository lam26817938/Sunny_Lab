import numpy as np
import cv2 as cv
import glob

chessboardSize = (9,6)
frameSize720p = (640, 360)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = []  # 2d points in image plane for left camera
imgpointsR = []  # 2d points in image plane for right camera

# Load images for both cameras
imagesLeft = glob.glob('CameraCalibration-main/images/LC/*.png')
imagesRight = glob.glob('CameraCalibration-main/images/RC/*.png')

if len(imagesLeft) != len(imagesRight):
    raise ValueError("Mismatched number of images between left and right cameras.")

# Process each pair of images
for imgLeftPath, imgRightPath in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeftPath)
    imgR = cv.imread(imgRightPath)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR:
        objpoints.append(objp)

        corners2L = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsL.append(corners2L)
        cv.drawChessboardCorners(imgL, chessboardSize, corners2L, retL)
        cv.imshow('Left Image', imgL)

        corners2R = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(corners2R)
        cv.drawChessboardCorners(imgR, chessboardSize, corners2R, retR)
        cv.imshow('Right Image', imgR)

        cv.waitKey(1000)

cv.destroyAllWindows()

############## CALIBRATION #######################################################

# 使用720p分辨率进行相机标定
retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize720p, None, None)
retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize720p, None, None)

# Stereo calibration
flags = cv.CALIB_FIX_INTRINSIC
criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ret, _, _, _, _, R, T, E, F = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, cameraMatrixL, distL, cameraMatrixR, distR, frameSize720p, criteria_stereo, flags)

# Stereo rectification
R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(cameraMatrixL, distL, cameraMatrixR, distR, frameSize720p, R, T)

# 生成用于720p分辨率的映射
mapLx, mapLy = cv.initUndistortRectifyMap(cameraMatrixL, distL, R1, P1, frameSize720p, cv.CV_32FC1)
mapRx, mapRy = cv.initUndistortRectifyMap(cameraMatrixR, distR, R2, P2, frameSize720p, cv.CV_32FC1)

# 对720p图像进行去畸变和校正
undistortedL = cv.remap(imgL, mapLx, mapLy, cv.INTER_LINEAR)
undistortedR = cv.remap(imgR, mapRx, mapRy, cv.INTER_LINEAR)

# Display the undistorted and rectified images
cv.imshow('Undistorted and Rectified Left Image', undistortedL)
cv.imshow('Undistorted and Rectified Right Image', undistortedR)

cv.waitKey(0)

# Save the stereo maps
print("saving")
cvfile = cv.FileStorage("CameraCalibration-main/StereoMap_360p.xml", cv.FILE_STORAGE_WRITE)
cvfile.write("StereoMapL_x", mapLx)
cvfile.write("StereoMapL_y", mapLy)
cvfile.write("StereoMapR_x", mapRx)
cvfile.write("StereoMapR_y", mapRy)
cvfile.release()