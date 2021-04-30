import pyrobot
import numpy as np

class Locobot:
    # ANG_HIDE_ARM = [1.8652060275016624, -0.2505083450577331, 0.9004433547235274, 0.9208613171291021, -0.007486316919657954]
    ANG_HIDE_ARM = [1.6210930191238981, 0.389896031387331, 0.7745488114879888, 0.40635148391957676, -0.0021963282894992275]
    RANGE_PAN = np.linspace(-0.7,0.7,10)
    RANGE_TILT = np.linspace(0.7,0.3,10)


    """ Controlling the Locobot through the pyrobot wrapper"""
    def __init__(self):
        self.bot = None
        self._initialize_locobot()
        # pan = left (+ve), right(-ve) / tilt = up (-ve), down(+ve)
        # self.bot.camera.reset()

        # quick fix for depthmap bug.
        self.bot.camera.depth_cam.cfg_data['DepthMapFactor']=1.0
        self.bot.camera.set_pan_tilt(0,0.7)
        self.bot.arm.go_home()
        self.camera_state = [0,0]
        self.camera_transform = pyrobot.algorithms.camera_transform.CameraTransform()

    def _initialize_locobot(self):
        self.bot = pyrobot.Robot("locobot", use_base=False)
        self.bot.arm.go_home()

    def _hover(self, target_pose):
        """ Move the hand above the 20cm above the given position"""
        hover_pose = target_pose["position"]
        hover_pose = hover_pose[2] + 0.2
        return self.bot.arm.set_ee_pose(**target_pose)

    def hide_arm(self):
        self.bot.arm.set_joint_positions(self.ANG_HIDE_ARM, plan=False)

    def camera_scan(self):
        # camera_state given in [pan, tilt]
        pan_state = self.camera_state[0]
        tilt_state = self.camera_state[1]
        self.bot.camera.set_pan_tilt(self.RANGE_PAN[pan_state], self.RANGE_TILT[tilt_state])

        self.camera_state[0] += 1
        if self.camera_state[0] == len(self.RANGE_PAN):
            self.camera_state[0] = 0
            self.camera_state[1] += 1
        if self.camera_state[1] == len(self.RANGE_TILT):
            self.camera_state[1] = 0 


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
        success = self.bot.arm.set_ee_pose(**target_pose)
        self.bot.gripper.close()
        return success

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
        self._hover(target_pose)
        self.bot.arm.set_ee_pose(**target_pose)
        self.bot.gripper.open()