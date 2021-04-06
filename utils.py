import numpy as np

def filter_points(pts, colors, z_lowest=0.01):
    valid = pts[:, 2] > z_lowest
    valid = np.logical_and(valid, pts[:, 0] < 0.5)
    valid = np.logical_and(valid, pts[:, 1] < 0.4)
    valid = np.logical_and(valid, pts[:, 1] > -0.4)
    pts = pts[valid]
    colors = colors[valid]
    return pts, colors