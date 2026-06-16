import open3d as o3d

def inside_pointcloud(points, colors, axis=False):
    # Construct and display the Open3D point cloud
    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    mid_z = (points[:, 2].min() + points[:, 2].max()) / 2

    if (axis == True):
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=[0, 0, 0]
        )

        o3d.visualization.draw_geometries(
            [pcd, axis],
            window_name="Edge Point Cloud Reconstruction",
            width=1200,
            height=800,
            lookat=[0, 0, mid_z],
            front=[0, 0, -1],
            up=[0, 1, 0],
            zoom=0.05
        )


    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Edge Point Cloud Reconstruction",
        width=1200,
        height=800,
        lookat=[0, 0, mid_z],
        front=[0, 0, -1],
        up=[0, 1, 0],
        zoom=0.05
    )


def origin_pointcloud(points, colors, axis=False):
    # Construct and display the Open3D point cloud
    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if (axis == True):
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=[0, 0, 0]
        )

        o3d.visualization.draw_geometries(
            [pcd, axis],
            # [pcd],
            window_name="RGB-D Point Cloud Reconstruction",
            width=1200,
            height=800,
            lookat=[0,0,0],
            front=[0,0,1],
            up=[0,1,0],
            zoom = 0.001
        )

    o3d.visualization.draw_geometries(
        [pcd],
        # [pcd],
        window_name="RGB-D Point Cloud Reconstruction",
        width=1200,
        height=800,
        lookat=[0,0,0],
        front=[0,0,1],
        up=[0,1,0],
        zoom = 0.001
    )
        
