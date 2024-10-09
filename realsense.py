import pyrealsense2 as rs
import numpy as np
import cv2


imW, imH = 1280, 720


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, imW, imH, rs.format.z16, 6)
config.enable_stream(rs.stream.color, imW, imH, rs.format.bgr8, 6)

# Start streaming
pipeline.start(config)

# Callback function for mouse click
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the depth value at the clicked point
        depth_value = depth_frame.get_distance(x, y)
        print(f"Depth at ({x}, {y}): {depth_value:.3f} meters")

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply a colormap to the depth image for better visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Display the images
        cv2.imshow('RealSense', images)

        # Set the mouse callback function
        cv2.setMouseCallback('RealSense', mouse_callback)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()