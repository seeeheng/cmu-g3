import pyrobot
import numpy as np
import signal
import argparse
import utils
import math
from LocoBlock import LocoBlock
# from PCLVisualizer import PCLVisualizer


# TODO: Shut down gracefully with opencv windows open.
# TODO: Figure out how to move arm out of frame. Then create a function for getting arm out of camera frame.

def main(args):
    n_tries = args.n_tries
    bot = LocoBlock()
    signal.signal(signal.SIGINT, bot.signal_handler)
    bot.do()

    # while True:
    #     # Simulating looking around
    #     bot.set_camera_pan_tilt([0,0.7])

    #     bot.pick(PLACEHOLDER_POSE_1)
    #     bot.place(PLACEHOLDER_POSE_2)

    # rgb, depth = bot.get_rgbd()
    # rgb_viz.update(rgb, depth)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for LoCoBot.")
    parser.add_argument(
        "--n_tries", help="Number of tries for pose estimation", type=int, default=5
    )
    args = parser.parse_args()
    main(args)
