import pyrobot

class Locobot:
    """ Controlling the Locobot through the pyrobot wrapper"""

    def __init__(self):
        self.bot = None
        self._initialize_locobot()

    def _initialize_locobot(self):
        self.bot = pyrobot.Robot("locobot", use_base=False)
        self.bot.arm.go_home()

    def _hover(self, target_pose):
        """ Move the hand above the 20cm above the given position"""
        hover_pose = target_pose.position
        hover_pose = hover_pose[2] + 0.2
        self.bot.arm.set_ee_pose(**target_pose)

    def get_rgbd(self):
        """ Returns RGB-D image
        
        Returns:
            rgb: rgb image in ndarray
            depth: depth image in ndarray
        """
        rgb, depth = self.bot.camera.get_rgb_depth()
        return rgb, depth

    def get_pcl(self, in_cam=False):
        """ Returns current PCL """
        pts, colors = self.bot.camera.get_current_pcd(in_cam=in_cam)
        return pts, colors
    
    def set_camera_pan_tilt(self, pose):
        """ Simple function for setting camera pan/tilt. """
        pan, tilt = pose
        self.bot.camera.set_pan_tilt(pan, tilt)

    def pick(self, target_pose):
        """ Given a pose, pick.

        Open gripper > plan EE to target_pose > close gripper.

        Inputs:
            target_pose: must be a dict with keys as "position" and "orientation".
            position must be a nd.array with 3 values.
            orientation can be either a nd.array with 4 values (Quaternions)
            or a nd.array with 3x3 values (Rotation Matrix.)
        """
        self.bot.gripper.open()
        self.bot.arm.set_ee_pose(**target_pose)
        self.bot.gripper.close()

    def place(self, target_pose):
        """ Given a target_pose, place.

        Plan EE to pose directly above target-pose > plan ee to target_pose > open gripper.

        Inputs:
            target_pose: must be a dict with keys as "position" and "orientation".
            position must be a nd.array with 3 values.
            orientation can be either a nd.array with 4 values (Quaternions)
            or a nd.array with 3x3 values (Rotation Matrix.)
        """
        self.bot.gripper.close()
        # might need interpolation -> point directly above, then lower the block down. Else will definitely hit the blocks.
        self._hover(**target_pose)
        self.bot.arm.set_ee_pose(**target_pose)
        self.bot.gripper.open()