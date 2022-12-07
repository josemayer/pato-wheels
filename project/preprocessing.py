import numpy as np
import cv2

# Color segmentation hyperspace
inner_lower = np.array([22, 93, 160])
inner_upper = np.array([45, 255, 255])
outer_lower = np.array([0, 0, 130])
outer_upper = np.array([179, 85, 255])

def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """ Returns a 2D array """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    inner_mask = cv2.inRange(hsv, inner_lower, inner_upper)
    outer_mask = cv2.inRange(hsv, outer_lower, outer_upper)
    return inner_mask, outer_mask
