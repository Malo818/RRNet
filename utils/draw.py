import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def draw_pic(data):
    data = np.asarray(data.cpu())
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, image in enumerate(data):
        row = i // 2
        col = i % 2
        axs[row, col].imshow(image)
        axs[row, col].axis('off')
    plt.tight_layout()
    plt.show()


def draw_point(data, color):

    data = np.asarray(data.cpu())
    color = np.asarray(color.cpu())

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data)

    colors = np.zeros((len(color), 3))
    colors[color == 1] = [1, 0, 0]

    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud])


def draw_point_nocolor(data):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([point_cloud])