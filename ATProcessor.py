import cv2
import dt_apriltags
import numpy as np

class ATProcessor:
    def __init__(self):
        # searchpath = config["searchpath"]
        # families = config['families']
        self.at_detector = dt_apriltags.Detector(
            searchpath=["apriltags"],
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
    
    def get_at_results(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = self.at_detector.detect(
            image_gray,
            estimate_tag_pose=True,
            camera_params=([1,1,1,1]),
            tag_size=0.015)
        return results
