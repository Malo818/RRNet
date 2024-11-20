from torch.utils.data import Dataset
import pickle

class PointCloudImageDataset(Dataset):
    def __init__(self, images, points, labels):
        assert len(images) == len(points)
        self.images = images
        self.points = points
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        point = self.points[idx]
        label = self.labels[idx]
        return image, point, label

def localdataset(dir):
    with open(dir, 'rb') as f:
        data = pickle.load(f)

    datasets = []
    for images, points, labels in zip(data['image'], data['point'], data['label']):
        dataset = PointCloudImageDataset(images, points, labels)
        datasets.append(dataset)

    return datasets
