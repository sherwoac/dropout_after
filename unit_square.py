import numpy as np
import torch


class UnitSquare(object):
    def __init__(self, dimensions: int):
        """
        container for pointyness
        :param dimensions: number of dimensions
        """
        self.dimensions = dimensions

    def contains_points(self, points: torch.Tensor) -> torch.Tensor:
      """
      unit square contains_points
      :param points: [nxp]
      :return: returns a bool Torch.Array of inside=True [p]
      """
      mask_below = points < torch.ones(size=(1, self.dimensions))
      mask_above = points > torch.zeros(size=(1, self.dimensions))
      mask_inside = mask_below * mask_above
      return mask_inside.all(dim=1)

    def cover_box(self, points, within_bounds_ratio):
        """
        helper function for covering box
        :param points:
        :param within_bounds_ratio:
        :return:
        """
        original_center = points.mean(dim=0)
        # centre on origin
        points -= original_center
        # expand box in ratio of 1/cubed root bounds
        points *= np.power(within_bounds_ratio, -1. / self.dimensions)
        # now give big box new center
        points += torch.tensor([[0.5, 0.5]])
        return points