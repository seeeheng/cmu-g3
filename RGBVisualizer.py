import cv2

class RGBVisualizer:
    """ For visualizing RGB."""
    def __init__(self):
        pass

    def update(self, rgb, depth):
        """ To be called in main thread at every iteration for updating the visualization.
       
        Args:
            rgb:
            depth:       
        """
        cv2.imshow("RGB", rgb[:, :, ::-1])
        cv2.imshow("Depth", 1000* depth)
        cv2.waitKey()