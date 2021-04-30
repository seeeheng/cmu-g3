import pyrobot
import numpy as np
import utils
import math
from Locobot import Locobot
# from PCLVisualizer import PCLVisualizer
from RGBVisualizer import RGBVisualizer
from ATProcessor import ATProcessor

# TODO: Shut down gracefully with opencv windows open.
# TODO: Figure out how to move arm out of frame. Then create a function for getting arm out of camera frame.

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
    # pcl_viz = PCLVisualizer()
    rgb_viz = RGBVisualizer()
    at_proc = ATProcessor()

    bot.hide_arm()

    frames = 0
    while True:
        rgb, depth = bot.get_rgbd()
        # pts, colors = bot.get_pcl()
        rgb_viz.update(rgb, depth)
        
        if frames == 500:
            bot.camera_scan()
            frames = 0
        else:
            frames += 1
        results = at_proc.get_at_results(rgb)
        if len(results) == 0:
            rgb_viz.show()
        for result in results:
            rgb_viz.annotate_polylines(result.corners)
            print("AT results pos={}, rot={}".format(result.pose_t, result.pose_R))

            print("Camera tf ={}".format(bot.bot.camera.get_link_transform("camera_color_optical_frame", "base_link")[2]))           
            
            # getting transformation from camera link to base link.
            camera_tf = bot.bot.camera.get_link_transform("camera_color_optical_frame", "base_link")[2]

            at_axes_transform = utils.rpyxyz2H((math.pi/2,math.pi/2,0),(0,0,0))
            at_transform = utils.rtmat2H(result.pose_R, result.pose_t)

            # rotate the results, then transform to robot base.
            true_t = np.matmul(np.matmul(at_transform, at_axes_transform),camera_tf)
            print("true_t={}".format(true_t))
            true_t_rotation = true_t[:3,:3]
            true_t_translation = true_t[:3,3]
            print("Calculated transpose={}, rotation={}".format(true_t_translation, true_t_rotation))
            pos = {
                "position": true_t_translation,
                "orientation": true_t_rotation
            }
            bot.pick(pos)
            exit()

            # # print(pos)
            # bot.pick({
            #     "position": np.array(result.pose_t),
            #     "orientation": np.array(result.pose_R)
            # })

    # while True:
    #     # Simulating looking around
    #     bot.set_camera_pan_tilt([0,0.7])

    #     bot.pick(PLACEHOLDER_POSE_1)
    #     bot.place(PLACEHOLDER_POSE_2)

    # rgb, depth = bot.get_rgbd()
    # rgb_viz.update(rgb, depth)
    


if __name__ == "__main__":
    main()
