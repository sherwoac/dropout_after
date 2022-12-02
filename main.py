import numpy as np
import torch
import matplotlib.pyplot as plt

import point_dataset
import point_classifier
import trainer


def plot_points(points, probabilities, title, filename=None):
  """
  plots the cube
  """
  colours = np.c_[(1 - probabilities), (1 - probabilities), np.zeros(shape=(probabilities.shape[0], 1))]
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.scatter(points[:, 0], points[:, 1], c=colours)
  ax.set_title(title)
  if filename is not None:
      plt.savefig(filename)

  plt.show()


if __name__ == '__main__':

    train_dataset = point_dataset.PointDataset.create_training_dataset()
    model_no_dropout = point_classifier.PointClassifierNoDropOut()
    model_trainer = trainer.Trainer(model_no_dropout)
    model_trainer.train(train_dataset, number_of_epochs=50)
    test_dataset = point_dataset.PointDataset.create_testing_dataset()
    no_drop_loss, no_drop_accuracy = model_trainer.validate(test_dataset)

    # plot me my square
    probs = model_no_dropout(test_dataset.data)
    plot_points(test_dataset.data, probs.squeeze().detach().numpy(),
                'occupancy probabilities for no dropout',
                'results/no_dropout.png')

    dropout_prob = 0.01
    model_dropout = point_classifier.PointClassifierDropout(model_no_dropout, dropout_probability=dropout_prob)
    model_trainer = trainer.Trainer(model_dropout)
    model_dropout.eval()
    probs = model_dropout(test_dataset.data)
    plot_points(test_dataset.data,
                probs.squeeze().detach().numpy(),
                f'occupancy probabilities for dropout p={dropout_prob}', 'results/dropout.png')

    number_of_trials = 1000
    for default_prob in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.]:
        model_dropout = point_classifier.PointClassifierDropout(model_no_dropout, dropout_probability=default_prob)
        model_trainer = trainer.Trainer(model_dropout)
        drop_loss, drop_accuracy = model_trainer.validate(test_dataset)
        print(f'{no_drop_loss.detach().numpy()=} '
              f'{drop_loss.detach().numpy()=} '
              f'{no_drop_accuracy.detach().numpy()=} '
              f'{drop_accuracy.detach().numpy()=}')

        # plot me my square
        probs_trials = np.zeros(shape=(number_of_trials, len(probs)))
        for i in range(number_of_trials):
            probs_trials[i] = model_dropout(test_dataset.data).squeeze().detach().numpy()

        std = probs_trials.std(axis=0)
        plot_points(test_dataset.data,
                    std,
                    f'std of occupancy probabilities for p={default_prob:.3f}',
                    f'results/dropout_{default_prob:.3f}.png')
