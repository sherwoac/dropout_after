import torch
from torch.utils.data.dataset import Dataset
import unit_square


class PointDataset(Dataset):
    """
    torch Dataset for sampling square
    """
    def __init__(self, points: torch.Tensor, square: unit_square.UnitSquare):
        """
        make a torch compatible dataset from points and a square
        :param points:
        :param square:
        """
        super(PointDataset, self).__init__()
        self.data = points
        self.labels = square.contains_points(points)

    @classmethod
    def create_training_dataset(cls, number_of_samples: int = 10000, dimensions: int = 2, training_coverage: float = 0.5):
        square = unit_square.UnitSquare(dimensions=dimensions)
        points = torch.rand(size=(number_of_samples, dimensions))
        spread_points = square.cover_box(points, training_coverage)
        return PointDataset(spread_points, square)

    @classmethod
    def create_testing_dataset(cls, dimensions: int = 2):
        a = torch.linspace(-0.499, 1.499, 20)
        x = torch.meshgrid([a]*dimensions, indexing="ij")
        evenly_spread_points = torch.vstack(list(map(torch.ravel, x))).T
        square = unit_square.UnitSquare(dimensions=dimensions)
        return PointDataset(evenly_spread_points, square)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
