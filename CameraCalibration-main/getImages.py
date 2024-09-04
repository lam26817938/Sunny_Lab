import cv2
import os

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

num = 0
print(os.getcwd())
folders = ['CameraCalibration-main/images/LC', 'CameraCalibration-main/images/RC']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")
        
while cap1.isOpened():

    succes1, img1 = cap1.read()
    succes2, img2 = cap2.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('CameraCalibration-main/images/LC/img' + str(num) + '.png', img1)
        cv2.imwrite('CameraCalibration-main/images/RC/img' + str(num) + '.png', img2)
        print("image saved!")
        num += 1

    cv2.imshow('Img1',img1)
    cv2.imshow('Img2',img2)

# Release and destroy all windows before termination
cap1.release()
cap2.release()

cv2.destroyAllWindows()