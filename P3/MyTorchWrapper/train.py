import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from .evaluation import BasicEvaluation
from .evaluation_results import EvaluationResults


class Trainer:
    """Class to train a Neural Network model."""

    seed_value = 10


    def __init__(
        self,
        model: nn.Module,
        evaluation: BasicEvaluation,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        device: torch.device,
    ) -> None:
        """
        Args:
            model (nn.Module): model to train
            evaluation (BasicEvaluation): evaluation instance with the desired
            methods of evaluation, including the loss. See the BasicEvaluation
            class for more details.
            epochs (int): number of training epochs
            optimizer (torch.optim.Optimizer): optimization algorithm to use
            data_loader (DataLoader): Data with which to train the torch model
            device (torch.device): device in which to perform the computations
        """
        self.model = model
        self.evaluation = evaluation
        self.epochs = epochs
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.device = device


    def train(self) -> EvaluationResults:
        """Train the torch model with the training data provided.

        Returns:
            EvaluationResults: Performance evaluation of the training process
            at each step.
        """
        torch.manual_seed(Trainer.seed_value)  # Ensure repeatable results
        self.model.to(self.device)
        self.model.train()  # Set the model in training mode

        total_steps = len(self.data_loader)
        feedback_step = round(total_steps / 3) + 1
        results = EvaluationResults()

        for epoch in range(self.epochs):
            # Iterate over all batches of the dataset
            for i, (features, labels) in enumerate(self.data_loader):
                # Move the data to the torch device
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)  # Forward pass
                loss = self.evaluation(outputs, labels, results)  # Evaluation

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) % feedback_step == 0 or i + 1 == total_steps:
                    print(
                        "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                            epoch + 1, self.epochs, i + 1, total_steps, loss.item()
                        )
                    )

        return results
