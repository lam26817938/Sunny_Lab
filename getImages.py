import cv2
import os

cap1 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


num = 22
print(os.getcwd())
folders = ['images/']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")
        
while cap1.isOpened():

    succes1, img1 = cap1.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/' + str(num) + '.png', img1)
        print("image saved!")
        num += 1

    cv2.imshow('Img1',img1)

# Release and destroy all windows before termination
cap1.release()

cv2.destroyAllWindows()