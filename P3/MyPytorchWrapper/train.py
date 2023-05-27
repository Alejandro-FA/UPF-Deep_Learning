import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
import torchinfo
from .evaluation import BasicEvaluation
from .test import Tester
from typing import Dict


class Trainer:
    """Class to train a Neural Network model.
    """    
    seed_value = 10

    def __init__(self, model: nn.Module, evaluation: BasicEvaluation, epochs: int, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        """_
        Args:
            model (nn.Module): model to train
            evaluation (BasicEvaluation): evaluation instance with the desired
            methods of evaluation, including the loss. See the BasicEvaluation
            class for more details.
            epochs (int): number of training epochs
            optimizer (torch.optim.Optimizer): optimization algorithm to use
            device (torch.device): device in which to perform the computations
        """        
        self.model = model
        self.evaluation = evaluation
        self.epochs = epochs
        self.optim = optimizer
        self.device = device


    def train(self, data_loader: DataLoader) -> Dict[str, float]:
        """Train the torch model with the training data provided.

        Args:
            data_loader (DataLoader): Data with which to train the torch model

        Returns:
            Dict[str, float]: Performance evaluation of the training process at
            each step.
        """        
        torch.manual_seed(Trainer.seed_value) # Ensure repeatable results
        self.model.to(self.device)
        self.model.train() # Set the model in training mode

        total_steps = len(data_loader)
        feedback_step = round(total_steps / 3) + 1

        for epoch in range(self.epochs):
            # Iterate over all batches of the dataset
            for i, (features, labels) in enumerate(data_loader):

                # Move the data to the torch device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features) # Forward pass
                loss = self.evaluation(outputs, labels) # Evaluation
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i+1) % feedback_step == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(epoch+1, self.epochs, i+1, total_steps, loss.item()))
                    
        return self.evaluation.get_results(only_last=False)
    

    def get_train_summary(self, data_loader: DataLoader) -> str:
        """Build a training summary report for future reference.

        Args:
            data_loader (DataLoader): Data used to train the model. Used to
            gather some addition information about the training process.

        Returns:
            str: Summary content. Usually used for printing it to screen or
            writing it to a file.
        """        
        batch, _ = next(iter(data_loader))

        tester = Tester(device=self.device, evaluation=self.evaluation.copy())
        results = tester.test(model=self.model, data_loader=data_loader)
        model_stats = torchinfo.summary(self.model, input_size=batch.shape, device=self.device, verbose=0)
        
        summary = (
            f"Training results: {results}"
            + f"Loss function used: {self.evaluation.loss_criterion}"
            + f"Epochs: {self.epochs}\n"
            + f"Optimizer: {self.optimizer}\n"
            + str(model_stats)
        )

        return summary
