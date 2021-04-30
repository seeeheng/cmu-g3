import socket
import time
from math import sqrt, pi
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation as R
from skspatial.objects import Plane, Points
import pyrealsense2 as rs
import matplotlib.pyplot as plt
#import cv2
from pyrobot.utils.util import try_cv2_import
cv2 = try_cv2_import()
import traceback

HOST = '10.0.2.50'
PORT = 30002
RNG = np.random.default_rng()
NUM_POINTS = 500
MIN_RADIUS = 20
MAX_RADIUS = 25

def connect_to_robot():
    s = socket.socket()
    s.connect((HOST,PORT))
    return s

def send_command(s,command):
    s.send(command)
    data = s.recv(1024)
    return data

def command_builder(point,a='0.25',v='0.35'):
    r_x = str(point[0] + 0.71)
    r_y = str(-point[2] - 0.42)
    r_z = str(point[1] + 0.64)
    phi = str(rotvec[0])
    eta = str(rotvec[1])
    zeta = str(rotvec[2])
    command_str = 'movej(p[' + r_x +',' + r_y + ',' + r_z + ',' + phi + ',' + eta +',' + zeta + '],a=' + a + ',v=' + v + ')\n'
    print(command_str)
    command = command_str.encode()
    return command

class CV_Pipe:

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Camera pipeline functions
    def start(self):
        print('Starting camera')
        profile = self.pipeline.start(self.config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        return

    def update_aligned_frames(self):
        # Wait for a coherent pair of frames: depth and color  
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        self.depth_frame = aligned_frames.get_depth_frame()
        self.color_frame = aligned_frames.get_color_frame()

        # Convert images to numpy arrays
        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.color_image = np.asanyarray(self.color_frame.get_data())

        self.intrin = self.depth_frame.profile.as_video_stream_profile().intrinsics
        return

    def output_image(self):
        # Apply colormap  on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = self.color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(self.color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((self.color_image, depth_colormap))

        return images

    # CV pipeline functions
    def detect_circles(self, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS):
        gray_img = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)    
        gray_img = cv2.GaussianBlur(gray_img, (7, 7), 1.5, 1.5);
        circles = cv2.HoughCircles(gray_img,cv2.HOUGH_GRADIENT,1,30,param1=25,param2=20,minRadius=minRadius,maxRadius=maxRadius)
        if circles is not None:
            print('Detected ' + str(circles.shape[1]) + ' circles')
            for i in circles[0,:]:
                i = i.astype(int)
                # draw the outer circle
                print(i[2])
                self.draw_circle((i[0], i[1]), i[2], (0,255,0), 2)
                # draw the center of the circle
                self.draw_circle((i[0], i[1]), 2, (0,0,255), 3)
            self.circles = circles
            self.points = circles[0, :, :-1]
            return circles.shape[1]
        else:
            print('No circles detected')
            return 0

    def update_real_points(self):
        real_points = []
        # Find real points
        for point in self.points:
            point = point.astype(int)
            print("real_points_loop")
            depth = self.depth_frame.as_depth_frame().get_distance(point[0], point[1])
            real_point = rs.rs2_deproject_pixel_to_point(self.intrin, [point[0], point[1]], depth)
            real_points.append(real_point)
            print(real_point)
        self.real_points = real_points

    def update_convex_hull(self):
        # Find convex hull of detected circles
        self.hull = ConvexHull(self.points)
        self.draw_polygon(self.hull.simplices)

    def update_delaunay(self):        
        # Break polygon down to triangles
        self.vertices = self.points[self.hull.vertices]
        self.tri = Delaunay(self.vertices)

    def gen_points_in_polygon(self, num_points, rng):
        sample_points = []
        for idx_simplex, simplex in enumerate(self.tri.simplices):
            tri_points = self.vertices[simplex].astype(int)
            self.draw_triangles(tri_points)
            sample_points.extend(self.gen_points_in_triangle(tri_points, idx_simplex, int(num_points/self.tri.simplices.size), rng))
        self.sample_points = sample_points

    def gen_points_in_triangle(self, tri_points, idx_simplex, num_points, rng):
        sample_points = []
        for i in range(num_points):
            too_close = False
            r1 = RNG.random()
            r2 = RNG.random()
            gen_point = ((1 - sqrt(r1)) * tri_points[0]) + ((sqrt(r1) * (1-r2)) * tri_points[1]) + ((r2 * sqrt(r1)) * tri_points[2])
            gen_point = gen_point.astype(int)
            for vertex in tri_points:
                if self.distance_two_points(vertex, gen_point) < MAX_RADIUS:
                    too_close = True
            if not too_close:
                self.draw_circle(tuple(gen_point), 2, (255,255,0), 1)
                depth = self.depth_frame.as_depth_frame().get_distance(gen_point[0], gen_point[1])
                point = rs.rs2_deproject_pixel_to_point(self.intrin, [gen_point[0], gen_point[1]], depth)
                sample_points.append(point)
        return sample_points

    def stop(self):
        print('Stopping camera')
        self.pipeline.stop()
        return

    # Spatial fuctions
    def best_fit_plane(self):
        self.plane = Plane.best_fit(Points(self.sample_points))
        return self.plane.normal

    def find_rotation(self):
        z_axis = [0,0,1]
        self.rot, err = R.align_vectors([0 - self.plane.normal], [z_axis])
        print('Euler angles')
        print(self.rot.as_euler('zyx',degrees=True))
        print('Rotation vector')
        print(self.rot.as_rotvec())


    # Drawing functions
    def draw_circle(self, point, radius, color=(255,255,0), thickness=1):
        cv2.circle(self.color_image, point, radius, color, thickness)

    def draw_polygon(self, simplices, color=(255,255,255), thickness=2):
        for simplex in simplices:
            start_point = tuple(self.points[simplex][0].astype(int))
            end_point = tuple(self.points[simplex][1].astype(int))
            cv2.line(self.color_image, start_point, end_point, color, thickness)

    def draw_triangles(self, tri_points, color=(255,255,255), thickness=2):
        cv2.line(self.color_image, tuple(tri_points[0]), tuple(tri_points[1]), color, thickness)
        cv2.line(self.color_image, tuple(tri_points[1]), tuple(tri_points[2]), color, thickness)
        cv2.line(self.color_image, tuple(tri_points[2]), tuple(tri_points[0]), color, thickness)

    # Utility functions
    def distance_two_points(self, point1, point2):
        return np.linalg.norm(point1-point2)

if __name__ == '__main__':
    try:
        pipe = CV_Pipe()
        pipe.start()
        
        num_circles = 0
        while num_circles != 4:
            pipe.update_aligned_frames()

            # Detect circles
            num_circles = pipe.detect_circles()

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', pipe.output_image())
            cv2.waitKey(1)


        pipe.update_real_points()
        pipe.update_convex_hull()
        pipe.update_delaunay()
        pipe.gen_points_in_polygon(NUM_POINTS, RNG)

        print('normal')
        pipe.best_fit_plane()
        pipe.best_fit_plane()
        print(pipe.plane.normal)
        pipe.find_rotation()

        rotvec=pipe.rot.as_rotvec()
        print('rotvec')
        print(rotvec)

        cv2.imshow('RealSense', pipe.output_image())
        cv2.waitKey(0)
        pipe.stop()

        # s = connect_to_robot()

        # # Move robot to each circle detected
        # for point in pipe.real_points:
        #     response = send_command(s,command_builder(point))
        #     time.sleep(10)
        #     print(response)

        # # Go home
        # response = send_command(s,command_builder([-0.35,0.35,0.5]))

    except Exception as e:
        print(e)
        print(traceback.format_exc())
        pipe.stop()
