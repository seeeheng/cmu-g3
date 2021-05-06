import numpy as np
import math

def filter_points(pts, colors, z_lowest=0.01):
    valid = pts[:, 2] > z_lowest
    valid = np.logical_and(valid, pts[:, 0] < 0.5)
    valid = np.logical_and(valid, pts[:, 1] < 0.4)
    valid = np.logical_and(valid, pts[:, 1] > -0.4)
    pts = pts[valid]
    colors = colors[valid]
    return pts, colors

def rtmat2H(r_mat, t_mat):
    """ Converts rotation matrix and transformation matrix to a homogenous transform. """
    T = np.eye(4)
    T[:3, :3] = r_mat
    T[0, 3] = t_mat[0]
    T[1, 3] = t_mat[1]
    T[2, 3] = t_mat[2]
    return T

def H2rtmat(H):
    return H[:3, :3], H[0:3, 3]

def rpyxyz2H(rpy, xyz):
    """ Constructs homogenous transform from rpy and xyz.
    """
    h_transpose = np.array([
        [1,0,0,xyz[0]],
        [0,1,0,xyz[1]],
        [0,0,1,xyz[2]],
        [0,0,0,1]
    ])

    h_rotx = np.array([
        [1,0,0,0],
        [0,math.cos(rpy[0]),-math.sin(rpy[0]),0],
        [0,math.sin(rpy[0]),math.cos(rpy[0]),0],
        [0,0,0,1]
    ])

    h_roty = np.array([
        [math.cos(rpy[1]),0,math.sin(rpy[1]),0],
        [0,1,0,0],
        [-math.sin(rpy[1]),0,math.cos(rpy[1]),0],
        [0,0,0,1]
    ])

    h_rotz = np.array([
        [math.cos(rpy[2]),-math.sin(rpy[2]),0,0],
        [math.sin(rpy[2]),math.cos(rpy[2]),0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])

    h_transform = np.matmul(np.matmul(np.matmul(h_transpose,h_rotx),h_roty),h_rotz)
    return h_transform

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])

    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])