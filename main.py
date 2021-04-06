import pyrobot
import numpy as np
from Locobot import Locobot
from PCLVisualizer import PCLVisualizer
from RGBVisualizer import RGBVisualizer

# only for testing script.
PLACEHOLDER_POSE_1 = {
    "position": np.array([0.789, 0.123, 0.456]),
    "orientation": np.array(
        [
            [0.5380200, -0.6650449, 0.5179283],
            [0.4758410, 0.7467951, 0.4646209],
            [-0.6957800, -0.0035238, 0.7182463],
        ]
    )
}
PLACEHOLDER_POSE_2 = {
    "position": np.array([0.123, 0.456, 0.789]),
    "orientation": np.array(
        [
            [0.5380200, -0.6650449, 0.5179283],
            [0.4758410, 0.7467951, 0.4646209],
            [-0.6957800, -0.0035238, 0.7182463],
        ]
    )
}

def main():
    bot = Locobot()
    pcl_viz = PCLVisualizer()
    rgb_viz = RGBVisualizer()

    while True:
        # Simulating looking around
        bot.set_camera_pan_tilt([0,0.7])

        bot.pick(PLACEHOLDER_POSE_1)
        bot.place(PLACEHOLDER_POSE_2)

        # Updating PCL visualizer.
        pts, colors = bot.get_pcl()
        pcl_viz.update(pts, colors)

        # Updating RGB visualizier
        rgb, depth = bot.get_rgbd()
        rgb_viz.update(rgb, depth)

if __name__ == "__main__":
    main()
