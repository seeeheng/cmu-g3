import cv2
import numpy as np

class RGBVisualizer:
    """ For visualizing RGB."""
    def __init__(self):
        self.rgb = None
        self.depth = None
        self.stopped = False
        pass

    def update(self, rgb, depth):
        """ To be called in main thread at every iteration for updating the visualization.
       
        Args:
            rgb:
            depth:       
        """
        self.rgb = rgb
        self.depth = depth
        # self.show()

    def show(self):
        if not self.stopped:
            cv2.imshow("RGB", self.rgb[:, :, ::-1])
            # cv2.imshow("Depth", 1000*self.depth)
            cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()

    def annotate_polylines(self, pts):
        """ To annotate """
        pts = pts.reshape((-1,1,2))
        cv2.polylines(self.rgb, np.int32([pts]), True, (255,0,0), 2)
        self.show()

    def exit(self):
        self.stopped = True
