import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Trainer:
    batch_size = 1000

    def __init__(self,
                 model: torch.nn.Module,
                 save_filename=None):
        self.model = model
        self.loss_function = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        self.save_filename = save_filename

    def train(self, dataset: torch.utils.data.Dataset, number_of_epochs: int = 100):
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=Trainer.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(number_of_epochs):
            with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    train_accuracy = ((outputs > 0.5).squeeze() == labels).float().mean()
                    loss = self.loss_function(outputs.squeeze(), labels.float())
                    loss.backward()
                    self.optimizer.step()
                    tepoch.set_postfix(ls=loss.item(), acc=100. * train_accuracy.item())

        if self.save_filename is not None:
            self.model._save(self.save_filename)

    def validate(self, testing_dataset: torch.utils.data.Dataset) -> (float, float):
        """
        run model against testing_dataset
        :param testing_dataset:
        :return: loss, accuracy
        """
        self.model.eval()
        test_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=Trainer.batch_size, shuffle=False)
        running_accuracy = 0.
        loss = 0.
        for inputs, labels in test_dataloader:
            outputs = self.model(inputs)
            running_accuracy += ((outputs > 0.5).squeeze() == labels).sum()
            loss += self.loss_function(outputs.squeeze(), labels.float())

        return loss/len(testing_dataset), running_accuracy/len(testing_dataset)



