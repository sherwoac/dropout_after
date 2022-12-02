import torch
import torch.nn as nn


class PointClassifier(nn.Module):
    layer_width = 20
    layers = 10

    def _save(self, save_filename: str):
        torch.save(self.sequential.state_dict(), save_filename)

    def _load(self, load_filename: str):
        state_dict = torch.load(load_filename)
        self.sequential.load_state_dict(state_dict)


class PointClassifierNoDropOut(PointClassifier):
    """
    PointClassifier: minimal binary classifier
    """
    def __init__(self, input_dimensions=2):
        super(PointClassifierNoDropOut, self).__init__()
        self.sequential = nn.Sequential()
        self.sequential.append(nn.Linear(input_dimensions, PointClassifier.layer_width))
        self.sequential.append(nn.ReLU())
        for _ in range(PointClassifier.layers - 2):
            self.sequential.append(nn.Linear(PointClassifier.layer_width,
                                             PointClassifier.layer_width))
            self.sequential.append(nn.ReLU())

        self.sequential.append(nn.Linear(PointClassifier.layer_width, 1))
        self.sequential.append(nn.Sigmoid())

    def forward(self, x):
        return self.sequential(x)


class PointClassifierDropout(PointClassifier):
    """
    PointClassifierDropout: copies the input but with dropout before each linear
    """
    def __init__(self, given_model: PointClassifier, dropout_probability: float = 0.6):
        super(PointClassifierDropout, self).__init__()
        self.sequential = nn.Sequential()
        for key, layer in given_model.sequential._modules.items():
            if not key == '0' and isinstance(layer, nn.Linear):
                self.sequential.append((nn.Dropout(dropout_probability)))
            self.sequential.append(layer)

    def forward(self, x):
        return self.sequential(x)

    def eval(self):
        self.sequential.eval()  # sets all layers to eval
        for layer in self.sequential:
            if isinstance(layer, nn.Dropout):
                layer.train()