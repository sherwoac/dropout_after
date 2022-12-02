import os.path
import unittest
from unittest import TestCase

import tempfile
import torch

import unit_square
import point_dataset
import point_classifier
import trainer


class TestDataset(TestCase):
    def test_points(self):
        number_of_samples = 1000
        points_xy = torch.rand(size=(number_of_samples, 2))
        square = unit_square.UnitSquare(dimensions=2)
        spread_points_xy = square.cover_box(points_xy, 0.5)
        print(torch.mean(spread_points_xy), torch.min(spread_points_xy), torch.max(spread_points_xy))
        print(square.contains_points(spread_points_xy).sum())

    def test_datasets(self):
        # length
        train_dataset = point_dataset.PointDataset.create_training_dataset(number_of_samples=1)
        self.assertTrue(len(train_dataset) == 1, f'{len(train_dataset)=} should be length 1')

        # outside
        train_dataset = point_dataset.PointDataset.create_training_dataset(training_coverage=0.0000001)
        data, label = train_dataset[0]
        self.assertFalse(label, f'{label=} should be false')

        # inside
        train_dataset = point_dataset.PointDataset.create_training_dataset(training_coverage=1.)
        data, label = train_dataset[0]
        self.assertTrue(label, f'{label=} should be true')
        self.assertTrue((data < torch.ones(size=(2, 1))).all(), f'{data=} should be below one')
        self.assertTrue((data > torch.zeros(size=(2, 1))).all(), f'{data=} should be above zero')

        # test quarter coverage
        test_dataset = point_dataset.PointDataset.create_testing_dataset()
        self.assertTrue(test_dataset.labels.sum() == 100, f'{test_dataset.labels.sum()=} should be 100.')
        self.assertTrue((test_dataset.labels == False).sum() == 300, f'{(test_dataset.labels == False).sum()} should be 300')


class TestTrainer(TestCase):
    def train_model(self, model, train_dataset):
        model_trainer = trainer.Trainer(model)
        model_trainer.train(train_dataset, number_of_epochs=100)

    def validate_trained_model(self, model, test_dataset):
        model_trainer = trainer.Trainer(model)
        loss, accuracy = model_trainer.validate(test_dataset)
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertTrue(loss > 0.)
        self.assertTrue(isinstance(accuracy, torch.Tensor))
        self.assertTrue(accuracy > 0.)
        return loss, accuracy

    def train_model_save(self, model, model_save_filename, train_dataset):
        self.train_model(model, train_dataset)
        model._save(model_save_filename)
        self.assertTrue(os.path.isfile(model_save_filename))

    def load_model_validate(self, model, model_save_filename, test_dataset):
        self.assertTrue(os.path.isfile(model_save_filename))
        model._load(model_save_filename)
        self.validate_trained_model(model, test_dataset)

    def test_train_nodropout_load_into_dropout_validate(self):
        train_dataset = point_dataset.PointDataset.create_training_dataset()

        model_no_dropout = point_classifier.PointClassifierNoDropOut()
        self.train_model(model_no_dropout, train_dataset)
        test_dataset = point_dataset.PointDataset.create_testing_dataset()
        nodrop_loss, nodrop_accuracy = self.validate_trained_model(model_no_dropout, test_dataset)

        # with no dropout prop, should be same result
        model_dropout = point_classifier.PointClassifierDropout(model_no_dropout, dropout_probability=0.)
        model_dropout.eval()
        drop_loss, drop_accuracy = self.validate_trained_model(model_dropout, test_dataset)
        self.assertTrue(drop_loss == nodrop_loss)
        self.assertTrue(drop_accuracy == nodrop_accuracy)

        # with all drop out should be awful
        model_dropout_1 = point_classifier.PointClassifierDropout(model_no_dropout, dropout_probability=1.)
        model_dropout_1.eval()
        drop_loss, drop_accuracy = self.validate_trained_model(model_dropout_1, test_dataset)
        self.assertFalse(drop_loss == nodrop_loss)
        self.assertFalse(drop_accuracy == nodrop_accuracy)
        self.assertAlmostEqual(drop_accuracy, 0.25)  # half Falses should be same, all Trues are wrong


if __name__ == '__main__':
    unittest.main()