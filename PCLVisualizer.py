import open3d

class PCLVisualizer:
    """ For visualizing pointclouds."""
    def __init__(self):
        self.vis = open3d.Visualizer()
        self.vis.create_window("3D Map")

        self.pcl = open3d.PointCloud()

        coord = open3d.create_mesh_coordinate_frame(1, [0, 0, 0])
        self.vis.add_geometry(self.pcl)
        self.vis.add_geometry(coord)

    def _open3d_update(self):
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

    def update(self, pts, colors):
        """ To be called in main thread at every iteration for updating the visualization.
        
        Args:
            pts:
            colors:       
        """
        self.pcl.clear()
        self.pcl.points = open3d.Vector3dVector(pts)
        self.pcl.colors = open3d.Vector3dVector(colors / 255.0)
        self._open3d_update()
