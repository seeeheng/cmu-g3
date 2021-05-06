import pyrobot
import numpy as np
from RGBVisualizer import RGBVisualizer
from ATProcessor import ATProcessor
import copy
from geometry_msgs.msg import Quaternion, PointStamped
import rospy
from tf import TransformListener
import tf
import time
import utils
from scipy.spatial.transform import Rotation as R

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
    HOME_POSITION = [0.0028849 , 0.04385043, 0.03886779, 0.99827757]
    N_RESET_ATTEMPTS = 5
    # N_FRAMES_SCAN = 500

    PREGRASP_HEIGHT = 0.2 # 0.2
    GRASP_HEIGHT = 0.13 # 0.13
    INSERTION_HEIGHT = 0.18
    BASE_FRAME = "base_link"
    KINECT_FRAME = "camera_color_optical_frame"
    GRIPPER_FRAME = "gripper_link"
    MAX_DEPTH = 3.0
    MIN_DEPTH = 0.1
    N_TRIES = 2
    SLEEP_TIME = 1
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

    def reset(self):
        success = False
        for _ in range(self.N_RESET_ATTEMPTS):
            success = self.bot.arm.set_joint_positions(self.RESET_POSITION, plan=False)
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
        # print("[debug] cur_depth = {}".format(cur_depth))
        z = self._get_z_mean(cur_depth, [pt[0], pt[1]])
        # print("[debug] depth of point is : {}".format(z))
        if z == 0.0:
            raise RuntimeError
        if norm_z is not None:
            z = z / norm_z
        u = pt[1]
        v = pt[0]
        # camera intrinsics P
        P = copy.deepcopy(self.bot.camera.camera_P)
        # print("[debug] P is: {}".format(P))
        P_n = np.zeros((3, 3))
        P_n[:, :2] = P[:, :2]
        P_n[:, 2] = P[:, 3] + P[:, 2] * z
        P_n_inv = np.linalg.inv(P_n)
        temp_p = np.dot(P_n_inv, np.array([u, v, 1]))
        temp_p = temp_p / temp_p[-1]
        temp_p[-1] = z
        return temp_p

    def _convert_frames(self, pt):
        assert len(pt) == 3
        # print("[debug] Point to convert: {}".format(pt))
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
        # print("[debug] Base point to convert: {}".format(base_pt))
        return base_pt

    def get_3D(self, pt, z_norm=None):
        """ 
        Get 3D point to grasp at, based on sampling using the depth camera.

        (Heavily inspired by LoCoBot example)
        https://github.com/facebookresearch/pyrobot/tree/master

        Args:
            pt: Point to grasp at.

        Returns:
            base_pt:
        """
        temp_p = self._get_3D_camera(pt, z_norm)
        base_pt = self._convert_frames(temp_p)
        print("[debug] base_pt now: {}".format(base_pt))
        return base_pt

    def set_pose(self, position, pitch=1.57, roll=0.0, plan=False):
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
                position=position, pitch=pitch, roll=roll, plan=plan, numerical=False
            )
            if success == 1:
                break
        return success

    def drop(self, drop_pose):
        self.bot.arm.go_home()
        pregrasp_position = [drop_pose[0], drop_pose[1], self.PREGRASP_HEIGHT+0.05]
        grasp_angle = drop_pose[2]
        grasp_position = [drop_pose[0], drop_pose[1], self.INSERTION_HEIGHT]

        drop_positions_x = np.arange(drop_pose[0]-0.005, drop_pose[0]+0.005, 0.001)
        drop_positions_y = np.arange(drop_pose[1]-0.005, drop_pose[1]+0.005, 0.001)
        drop_heights = np.arange(self.INSERTION_HEIGHT+0.02, self.INSERTION_HEIGHT-0.02,-0.04/10)
        drop_execute_list = list(zip(drop_positions_x, drop_positions_y, drop_heights))

        # result = self.set_pose([drop_pose[0]-0.005, drop_pose[1]-0.005, self.INSERTION_HEIGHT+0.04], roll=grasp_angle, plan=False)
        # if not result:
        #     return False
        # time.sleep(self.SLEEP_TIME)
        
        # import math
        # result = self.set_pose([drop_pose[0]+0.005, drop_pose[1]+0.005, self.INSERTION_HEIGHT], roll=grasp_angle, plan=True)
        # if not result:
        #     return False
        # time.sleep(self.SLEEP_TIME)

        for trajectory in drop_execute_list:
            print(trajectory)
            result = self.set_pose(trajectory, roll=grasp_angle)
            if not result:
                return False
            time.sleep(self.SLEEP_TIME)

        # rospy.loginfo("Going to pre-drop pose:\n\n {} \n".format(pregrasp_position))
        # result = self.set_pose(pregrasp_position, roll=grasp_angle+math.pi/10, plan=True)
        # if not result:
        #     return False
        # time.sleep(self.SLEEP_TIME)


        # rospy.loginfo("Going to grasp pose:\n\n {} \n".format(grasp_position))
        # result = self.set_pose(grasp_position, roll=grasp_angle, plan=True)
        # if not result:
        #     return False
        # time.sleep(self.SLEEP_TIME)

        rospy.loginfo("Opening gripper")
        self.bot.gripper.open()
        return True

    def pre_grasp(self, grasp_pose):
        self.bot.arm.go_home()
        pregrasp_position = [grasp_pose[0], grasp_pose[1], self.PREGRASP_HEIGHT]
        rospy.loginfo("Going to pre-grasp pose:\n\n {} \n".format(pregrasp_position))
        result = self.set_pose(pregrasp_position)
        if not result:
            return False
        time.sleep(self.SLEEP_TIME)
        return True

    def grasp(self, grasp_pose):
        self.bot.arm.go_home()
        pregrasp_position = [grasp_pose[0], grasp_pose[1], self.PREGRASP_HEIGHT]
        rospy.loginfo("Going to pre-grasp pose:\n\n {} \n".format(pregrasp_position))
        result = self.set_pose(pregrasp_position)
        if not result:
            return False
        time.sleep(self.SLEEP_TIME)

        # pregrasp_pose = Pose(Point(*pregrasp_position), self.default_Q)
        import math
        # TODO: how???
        grasp_angle = 0
        # grasp_angle = grasp_pose[2]-math.pi/2
        # if grasp_angle > math.pi:
        #     grasp_angle -= math.pi
        print("[debug_angle] ANGLE={}".format(grasp_angle))

        # grasp_angle = self.get_grasp_angle(grasp_pose)
        grasp_position = [grasp_pose[0], grasp_pose[1], self.GRASP_HEIGHT]
        # grasp_pose = Pose(Point(*grasp_position), self.grasp_Q)

        rospy.loginfo("Going to grasp pose:\n\n {} \n".format(grasp_position))
        result = self.set_pose(grasp_position, roll=grasp_angle)
        if not result:
            return False
        time.sleep(self.SLEEP_TIME)

        rospy.loginfo("Closing gripper")
        self.bot.gripper.close()
        time.sleep(self.SLEEP_TIME)
        gripper_state = self.bot.gripper.get_gripper_state()
        print("GRIPPER STATE = {}".format(gripper_state))
        if gripper_state != 2:
            self.bot.gripper.open()
            return False

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
            
            # if self.frames_processed == self.N_FRAMES_SCAN:
            #     self.camera_scan()
            #     self.frames_processed = 0
            # else:
            #     self.frames_processed += 1
            results = self.at_proc.get_at_results(rgb)
            if len(results) == 0:
                self.rgb_viz.show()
            
            # grasping
            if self.current_stage == 0:
                for result in results:
                    if result.tag_id == 0:
                        self.rgb_viz.annotate_polylines(result.corners)
                        # print("AT results={}".format(result))

                        ## ATTEMPT 1: some rotation.
                        # import math
                        # rotation_matrix = utils.rotationMatrixToEulerAngles(result.pose_R)
                        # print("R={}".format(rotation_matrix))
                        # print("Rdeg={}".format(
                        #     np.array([
                        #         math.degrees(rotation_matrix[0]),
                        #         math.degrees(rotation_matrix[1]),
                        #         math.degrees(rotation_matrix[2])])
                        #     )
                        # )
                        
                        ## ATTEMPT 2: get rotation from base_frame. Realized it doesn't make sense.
                        # t,r = self._transform_listener.lookupTransform(self.BASE_FRAME, self.KINECT_FRAME, rospy.Time(0))
                        # r = R.from_quat(r)
                        # rotation_matrix = np.matmul(r.as_matrix(),result.pose_R)
                        # rotation_matrix = utils.rotationMatrixToEulerAngles(rotation_matrix)

                        # import math
                        # print("Rdeg={}".format(
                        #     np.array([
                        #         math.degrees(rotation_matrix[0]),
                        #         math.degrees(rotation_matrix[1]),
                        #         math.degrees(rotation_matrix[2])])
                        #     )
                        # )
                                                
                        pix_center = [result.center[1],result.center[0]]
                        
                        # expects in height, then width. AKA y, then x.
                        base_pt = self.get_3D(pix_center)[:2]
                        # Adding on the rotation, 0 first.
                        base_pt = np.append(base_pt, 0)
                        # success = self.pre_grasp(base_pt)
                        # if success:
                            # at_r = R.from_matrix(result.pose_R)
                            # at_quaternion = at_r.as_quat()
                            # from geometry_msgs.msg import PoseStamped
                            # ps = PoseStamped()
                            # ps.header.frame_id = self.KINECT_FRAME
                            # ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = result.pose_t
                            # ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = at_quaternion 
                            # base_ps = self._transform_listener.transformPose(self.GRIPPER_FRAME, ps)
                            # print(base_ps)
                            # import math
                            # t,r = self._transform_listener.lookupTransform(self.GRIPPER_FRAME, self.KINECT_FRAME, rospy.Time(0))
                            # r = R.from_quat(r) # convert from quaternion to rotation matrix
                            # gripper2kinect_tf = utils.rtmat2H(r.as_matrix(), t)
                            # kinect2at_tf = utils.rtmat2H(result.pose_R, result.pose_t)
                            # at_tf = utils.rpyxyz2H([-math.pi/2,math.pi/2,0],[0,0,0])
                            # kinect2at_tf = np.matmul(kinect2at_tf, at_tf)
                            # gripper2at_tf = np.matmul(gripper2kinect_tf, kinect2at_tf)
                            # gripper2at_r, gripper2at_t = utils.H2rtmat(gripper2at_tf)
                            # print("gripper2at_tf = {}".format(gripper2at_tf))
                            # print("gripper2at_rads={}".format(utils.rotationMatrixToEulerAngles(gripper2at_r)))
                            # print("gripper2at_angles=roll{},pitch{},yaw{}".format(
                            #     math.degrees(utils.rotationMatrixToEulerAngles(gripper2at_r)[0]),
                            #     math.degrees(utils.rotationMatrixToEulerAngles(gripper2at_r)[1]),
                            #     math.degrees(utils.rotationMatrixToEulerAngles(gripper2at_r)[2]),
                            #     ))
                            # input()
                            # self.hide_arm()
                            # continue


                        success = self.grasp(base_pt)
                        if success:
                            # self.bot.gripper.open()
                            self.hide_arm()
                            self.current_stage += 1
                        else:
                            self.hide_arm()

            if self.current_stage == 1:
                for result in results:
                    if result.tag_id == 3:
                        self.rgb_viz.annotate_polylines(result.corners) 
                        print("AT results={}".format(result))  
                        pix_center = [result.center[1],result.center[0]]
                        base_pt = self.get_3D(pix_center)[:2]
                        # rotation_matrix = utils.rotationMatrixToEulerAngles(result.pose_R)
                        # print("Rotation: {}".format(rotation_matrix))
                        import math
                        base_pt = np.append(base_pt, math.pi/2)
                        success = self.drop(base_pt)
                        if success:
                            self.bot.arm.go_home()
                            self.hide_arm()
                            self.current_stage += 1
                        else:
                            self.bot.arm.go_home()
                            self.hide_arm()
            
            if self.current_stage == 3:
                x = input()
                if x == "y":
                    self.current_stage = 0


    def exit(self):
        """ Graceful exit, adpated from LoCoBot example. """
        self.reset()
        exit(0)

    def signal_handler(self, sig, frame):
        """ Signal handling function, adapted from LoCoBot example. """
        print("Exit called.")
        self.exit()

                        # fixed_pose_t = [result.pose_t[0],result.pose_t[2],result.pose_t[1]]
                        # at_transform = utils.rtmat2H(result.pose_R, fixed_pose_t)
                        # _,_,transform = self.bot.camera.get_link_transform(self.KINECT_FRAME, self.BASE_FRAME)
                        # print("APRILTAG transform = {}".format(at_transform))
                        # print("camera-base transform = {}".format(transform))
                        # converted_trf = np.matmul(at_transform, transform)
                        # rotation, translation = utils.H2rtmat(converted_trf)
                        # print("[HT debug] calculated rotation={}, calculated_translation={}".format(rotation, translation))
                        # self.bot.arm.set_ee_pose(
                        #     position = translation,
                        #     orientation = rotation,
                        #     plan = True,
                        #     wait = True,
                        #     numerical = False
                        # )
