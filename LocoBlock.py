import pyrobot
import numpy as np
from RGBVisualizer import RGBVisualizer
from ATProcessor import ATProcessor
import copy
from geometry_msgs.msg import Quaternion, PointStamped
import rospy
from tf import TransformListener
import time

# TODO: Consider

class LocoBlock:
    """ Robot that does block things """
    # ANG_HIDE_ARM = [1.8652060275016624, -0.2505083450577331, 0.9004433547235274, 0.9208613171291021, -0.007486316919657954]
    ANG_HIDE_ARM = [1.6210930191238981, 0.389896031387331, 0.7745488114879888, 0.40635148391957676, -0.0021963282894992275]
    RANGE_PAN = np.linspace(-0.7,0.7,10)
    RANGE_TILT = np.linspace(0.7,0.3,10)
    RESET_PAN = 0.0
    RESET_TILT = 0.8
    RESET_POSITION = [-1.5, 0.5, 0.3, -0.7, 0.0]
    N_RESET_ATTEMPTS = 5
    N_FRAMES_SCAN = 500
    PREGRASP_HEIGHT = 0.2 # 0.2
    GRASP_HEIGHT = 0.13 # 0.13
    BASE_FRAME = "base_link"
    KINECT_FRAME = "camera_color_optical_frame"
    MAX_DEPTH = 3.0
    MIN_DEPTH = 0.1
    N_TRIES = 2
    SLEEP_TIME = 2
    STAGES = ["grasp","insert"]    

    BB_SIZE = 5

    """ Controlling the Locobot through the pyrobot wrapper"""
    def __init__(self):
        self.bot = None
        self._initialize_locobot()
        # pan = left (+ve), right(-ve) / tilt = up (-ve), down(+ve)
        # self.bot.camera.reset()

        # quick fix for depthmap bug.
        # self.bot.camera.depth_cam.cfg_data['DepthMapFactor']=1.0
        self.bot.camera.set_pan_tilt(0,0.7)
        self.bot.arm.go_home()
        self.camera_state = [0,0]
        self.rgb_viz = RGBVisualizer()
        self.at_proc = ATProcessor()
        self.frames_processed = 0
        self._transform_listener = TransformListener()
        self.current_stage = 0

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
        success = self.bot.arm.set_ee_pose(**target_pose, numerical=False, plan=True)
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

    def reset(self):
        success = False
        for _ in range(self.N_RESET_ATTEMPTS):
            success = self.bot.arm.set_joint_positions(self.RESET_POSITION)
            if success == True:
                break
        self.bot.gripper.open()
        self.bot.camera.set_pan(self.RESET_PAN)
        self.bot.camera.set_tilt(self.RESET_TILT)
        return success

    def _process_depth(self, cur_depth=None):
        if cur_depth is None:
            cur_depth = self.bot.camera.get_depth()
        # cur_depth = cur_depth / 1000.0  # conversion from mm to m NOTE: This is not needed anymore since commit 9ea22bd
        print("cur_depth max={}, cur_depth min={}".format(cur_depth.max(), cur_depth.min()))
        cur_depth[cur_depth > self.MAX_DEPTH] = 0.0
        return cur_depth

    def _get_z_mean(self, depth, pt, bb=5):
        sum_z = 0.0
        nps = 0
        for i in range(bb * 2):
            for j in range(bb * 2):
                new_pt = [pt[0] - bb + i, pt[1] - bb + j]
                try:
                    new_z = depth[int(new_pt[0]), int(new_pt[1])]
                    if new_z > self.MIN_DEPTH:
                        sum_z += new_z
                        nps += 1
                except:
                    pass
        if nps == 0.0:
            return 0.0
        else:
            return sum_z / nps

    def _get_3D_camera(self, pt, norm_z=None):
        assert len(pt) == 2
        cur_depth = self._process_depth()
        print("[debug] cur_depth = {}".format(cur_depth))
        z = self._get_z_mean(cur_depth, [pt[0], pt[1]])
        print("[debug] depth of point is : {}".format(z))
        if z == 0.0:
            raise RuntimeError
        if norm_z is not None:
            z = z / norm_z
        u = pt[1]
        v = pt[0]
        # camera intrinsics P
        P = copy.deepcopy(self.bot.camera.camera_P)
        print("[debug] P is: {}".format(P))
        P_n = np.zeros((3, 3))
        P_n[:, :2] = P[:, :2]
        P_n[:, 2] = P[:, 3] + P[:, 2] * z
        P_n_inv = np.linalg.inv(P_n)
        temp_p = np.dot(P_n_inv, np.array([u, v, 1]))
        temp_p = temp_p / temp_p[-1]
        temp_p[-1] = z
        return temp_p

    def get_3D(self, pt, z_norm=None):
        """ Heavily inspired by LoCoBot example """
        temp_p = self._get_3D_camera(pt, z_norm)
        # print("temp_p: {}".format(temp_p))
        print("[debug] temp_p obtained = {}".format(temp_p))
        base_pt = self._convert_frames(temp_p)
        # HARD CODED.
        # base_pt = base_pt/-100
        # HARD CODED.
        print("[debug] base_pt now: {}".format(base_pt))
        return base_pt
        # return temp_p

    def _convert_frames(self, pt):
        assert len(pt) == 3
        print("[debug] Point to convert: {}".format(pt))
        ps = PointStamped()
        ps.header.frame_id = self.KINECT_FRAME
        ps.point.x, ps.point.y, ps.point.z = pt
        base_ps = self._transform_listener.transformPoint(self.BASE_FRAME, ps)
        print(
            "transform : {}".format(
                self._transform_listener.lookupTransform(
                    self.BASE_FRAME, self.KINECT_FRAME, rospy.Time(0)
                )
            )
        )
        base_pt = np.array([base_ps.point.x, base_ps.point.y, base_ps.point.z])
        print("[debug] Base point to convert: {}".format(base_pt))
        return base_pt

    def get_grasp_angle(self, grasp_pose):
        """ 
        Obtain normalized grasp angle from the grasp pose.

        This is needs since the grasp angle is relative to the end effector.
        
        :param grasp_pose: Desired grasp pose for grasping.
        :type grasp_pose: list

        :returns: Relative grasp angle
        :rtype: float
        """

        cur_angle = np.arctan2(grasp_pose[1], grasp_pose[0])
        delta_angle = grasp_pose[2] + cur_angle
        if delta_angle > np.pi / 2:
            delta_angle = delta_angle - np.pi
        elif delta_angle < -np.pi / 2:
            delta_angle = 2 * np.pi + delta_angle
        return delta_angle

    def set_pose(self, position, pitch=1.57, roll=0.0):
        """ 
        Sets desired end-effector pose.
        
        :param position: End-effector position to reach.
        :param pitch: Pitch angle of the end-effector.
        :param roll: Roll angle of the end-effector

        :type position: list
        :type pitch: float
        :type roll: float

        :returns: Success of pose setting process.
        :rtype: bool
        """

        success = 0
        for _ in range(self.N_TRIES):
            position = np.array(position)
            success = self.bot.arm.set_ee_pose_pitch_roll(
                position=position, pitch=pitch, roll=roll, plan=False, numerical=True
            )
            if success == 1:
                break
        return success

    def drop(self, drop_pose):
        self.bot.arm.go_home()
        pregrasp_position = [drop_pose[0], drop_pose[1], self.PREGRASP_HEIGHT]
        # pregrasp_pose = Pose(Point(*pregrasp_position), self.default_Q)
        grasp_angle = self.get_grasp_angle(drop_pose)
        grasp_position = [drop_pose[0], drop_pose[1], self.GRASP_HEIGHT]
        # grasp_pose = Pose(Point(*grasp_position), self.grasp_Q)

        rospy.loginfo("Going to pre-grasp pose:\n\n {} \n".format(pregrasp_position))
        result = self.set_pose(pregrasp_position, roll=grasp_angle)
        if not result:
            return False
        time.sleep(self.SLEEP_TIME)

        rospy.loginfo("Opening gripper")
        self.bot.gripper.open()
        return True

    def grasp(self, grasp_pose):
        self.bot.arm.go_home()
        pregrasp_position = [grasp_pose[0], grasp_pose[1], self.PREGRASP_HEIGHT]
        # pregrasp_pose = Pose(Point(*pregrasp_position), self.default_Q)
        grasp_angle = self.get_grasp_angle(grasp_pose)
        grasp_position = [grasp_pose[0], grasp_pose[1], self.GRASP_HEIGHT]
        # grasp_pose = Pose(Point(*grasp_position), self.grasp_Q)

        rospy.loginfo("Going to pre-grasp pose:\n\n {} \n".format(pregrasp_position))
        result = self.set_pose(pregrasp_position, roll=grasp_angle)
        if not result:
            return False
        time.sleep(self.SLEEP_TIME)

        rospy.loginfo("Going to grasp pose:\n\n {} \n".format(grasp_position))
        result = self.set_pose(grasp_position, roll=grasp_angle)
        if not result:
            return False
        time.sleep(self.SLEEP_TIME)

        rospy.loginfo("Closing gripper")
        self.bot.gripper.close()
        time.sleep(self.SLEEP_TIME)

        rospy.loginfo("Going to pre-grasp pose")
        result = self.set_pose(pregrasp_position, roll=grasp_angle)
        if not result:
            return False
        time.sleep(self.SLEEP_TIME)

        # rospy.loginfo("Opening gripper")
        # self.bot.gripper.open()
        return True
        # time.sleep(self.SLEEP_TIME)


    def do(self):
        self.hide_arm()

        while True:
            rgb, depth = self.get_rgbd()
            self.rgb_viz.update(rgb, depth)
            
            if self.frames_processed == self.N_FRAMES_SCAN:
                self.camera_scan()
                self.frames_processed = 0
            else:
                self.frames_processed += 1
            results = self.at_proc.get_at_results(rgb)
            if len(results) == 0:
                self.rgb_viz.show()
            
            # grasping
            if self.current_stage == 0:
                for result in results:
                    if result.tag_id == 0:
                        self.rgb_viz.annotate_polylines(result.corners)
                        print("AT results={}".format(result))
                        pix_center = result.center

                        base_pt = self.get_3D(pix_center)

                        self.grasp(base_pt)
                        # todo: if pass
                        self.current_stage += 1

            if self.current_stage == 1:
                for result in results:
                    if result.tag_id == 3:
                        self.rgb_viz.annotate_polylines(result.corners)   
                        pix_center = result.center
                        base_pt = self.get_3D(pix_center)
                        self.drop(base_pt)
                        exit()


    def exit(self):
        """ Graceful exit, adpated from LoCoBot example. """
        self.reset()
        exit(0)

    def signal_handler(self, sig, frame):
        """ Signal handling function, adapted from LoCoBot example. """
        print("Exit called.")
        self.exit()