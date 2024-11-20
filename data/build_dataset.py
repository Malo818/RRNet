import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree



class MultiViewPointCloudDataset(data.Dataset):
    def __init__(self, file_path, d_point, imagesize, point_density, N_view=4):
        self.file_path = file_path
        self.d_point = d_point
        self.imagesize = imagesize
        self.point_density = point_density
        self.N_view = N_view

        self.rawdata, self.density = self._load_data()
        self.images, self.imgpots = self._build_image()
        self.points = self._build_point(len(self.images))
        self.labels = self._build_label()

    def _load_data(self):
        point_cloud = o3d.io.read_point_cloud(self.file_path)

        cl, ind = point_cloud.remove_radius_outlier(nb_points=200, radius=0.1)  # Remove outliers
        inlier_cloud = point_cloud.select_by_index(ind)

        points = np.asarray(inlier_cloud.points)                                # Center point correction
        ct_points = self._coordinate_transformation(points)
        colors = np.asarray(inlier_cloud.colors)
        rawdata = np.concatenate((ct_points, colors), axis=1)                  # （N，6）
        density = self._density_calculation(ct_points)
        return rawdata, density

    def _build_image(self):
        sampled_points = self._weighted_random_sampling(self.rawdata)
        imgpots = []
        images = []
        for single_point in sampled_points:
            x, y, z = single_point[:3]
            min_x, max_x = x - self.d_point/2, x + self.d_point/2
            min_y, max_y = y - self.d_point/2, y + self.d_point/2
            min_z, max_z = z - self.d_point/2, z + self.d_point/2
            mask = (self.rawdata[:, 0] >= min_x) & (self.rawdata[:, 0] <= max_x) & \
                   (self.rawdata[:, 1] >= min_y) & (self.rawdata[:, 1] <= max_y) & \
                   (self.rawdata[:, 2] >= min_z) & (self.rawdata[:, 2] <= max_z)
            result_points = self.rawdata[mask]

            imgpots.append(result_points[:, :3])
            image = self._draw_image(single_point, result_points)
            images.append(image)
        return images, imgpots

    def _build_point(self, num):
        points = []
        dspoints = self._voxel_subsampling(self.rawdata, 4*self.density)
        density = self._density_calculation(dspoints)
        sam_N = int(self.point_density/density)

        for i in range(num):
            dspoints_copy = np.copy(dspoints).astype(np.float32)

            np.random.shuffle(dspoints_copy)
            point = []
            for j in range(sam_N):
                point_num = len(dspoints)//sam_N
                point.append(dspoints_copy[j*point_num: (j+1)*point_num])
            points.append(point)
        return points

    def _build_label(self):
        labels= []
        for i in range(len(self.points)):
            label = []
            for j in range(len(self.points[0])):
                points_cloud = self.points[i][j]
                imgpot_cloud = self.imgpots[i]

                tree = cKDTree(imgpot_cloud)
                min_distances, _ = tree.query(points_cloud)

                lab = (min_distances < self.density).astype(int)
                label.append(lab)
            labels.append(label)
        return labels

    def _voxel_subsampling(self, data, voxel_size):
        points = data[:, :3]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        dspoints = point_cloud.voxel_down_sample(voxel_size)
        onpoints = np.asarray(dspoints.points)
        return onpoints

    def _draw_image(self, single_point, data):
        h = self.imagesize
        w = self.imagesize
        color = data[:, 3:]
        images = []

        y_values = self.rawdata[:, 1]
        single_point_y = single_point[1]
        y_min = np.min(y_values)
        y_max = np.max(y_values)
        y_range = y_max - y_min
        lower_threshold = y_min + y_range / 3
        upper_threshold = y_min + 2 * y_range / 3
        if single_point_y <= lower_threshold:
            theta_x = np.radians(45)
        elif lower_threshold < single_point_y <= upper_threshold:
            theta_x = np.radians(0)
        else:
            theta_x = np.radians(-45)

        for i in range(self.N_view):
            theta_y = np.radians(360/self.N_view)*i

            rotation_matrix_y = np.array([
                [np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]
            ])
            rotation_matrix_x = np.array([
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]
            ])

            # Rotate the matrix
            data_y = np.dot(data[:, 0:3], rotation_matrix_y.T)
            data_yx = np.dot(data_y, rotation_matrix_x.T)
            data_image = np.concatenate((data_yx, color), axis=1)

            w_min = np.min(data_image[:, 0])
            w_max = np.max(data_image[:, 0])
            h_min = np.min(data_image[:, 1])
            h_max = np.max(data_image[:, 1])
            w_colist = np.linspace(w_min, w_max, w)
            h_colist = np.linspace(h_min, h_max, h)

            image = np.zeros((h, w, 3), dtype=np.float32)
            image_deep = np.zeros((h, w), dtype=np.float32)

            for dat in data_image:
                w_index = (np.abs(w_colist - dat[0])).argmin()
                h_index = (np.abs(h_colist - dat[1])).argmin()

                if image_deep[h - 1 - h_index, w_index] == 0:
                    image[h - 1 - h_index, w_index] = dat[3:]
                    image_deep[h - 1 - h_index, w_index] = dat[2]
                elif dat[2] > image_deep[h - 1 - h_index, w_index]:
                    image[h - 1 - h_index, w_index] = dat[3:]
                    image_deep[h - 1 - h_index, w_index] = dat[2]

            images.append(image)
        return images

    def _weighted_random_sampling(self, data):
        coordinates = data[:, 0:3]
        nbrs = NearestNeighbors(n_neighbors=20, algorithm='auto').fit(coordinates)
        distances, _ = nbrs.kneighbors(coordinates)
#       densities = np.power(np.exp(1.0 / (np.mean(distances, axis=1) + 1e-8)),1/5)   # Weight scaling
        densities = 1.0 / (np.mean(distances, axis=1) + 1e-8)
        weights = densities / np.sum(densities)
        num_samples =  32 #int(25*np.prod(np.max(coordinates, axis=0)-np.min(coordinates, axis=0))/(self.d_point**3))  Select the number of sampling points based on the scene volume
        sampled_indices = np.random.choice(len(data), size=num_samples, replace=False, p=weights)
        sampled_points = data[sampled_indices]
        return sampled_points

    def _coordinate_transformation(self, data):
        center = np.mean(data, axis=0)
        translation_vector = -center
        translated_points = data + translation_vector
        return translated_points

    def _density_calculation(self, data):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(data[:, :3])
        distances = point_cloud.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        return avg_dist
















