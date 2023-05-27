import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from .evaluation import Evaluation
from typing import Dict


class Tester:
    """Class to evaluate the performance of a trained model.
    """    

    def __init__(self, model: nn.Module, evaluation: Evaluation, device: torch.device) -> None:
        """
        Args:
            model (nn.Module): the model to test
            device (torch.device): device in which to perform the computations
            evaluation (Evaluation): evaluation instance with the desired
            methods of evaluation, including the loss. See the Evaluation class
            for more details.
        """        
        self.model = model
        self.evaluation = evaluation
        self.device = device


    def test(self, data_loader: DataLoader) -> Dict[str, float]:
        """Test the performance of a model with a given dataset.

        Args:
            data_loader (DataLoader): the dataset to test the model with.

        Returns:
            Dict[str, float]: Performance results. See the Evaluation class for
            more details.
        """
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for features, labels in data_loader:
                # Move the data to the torch device
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass (network predictions)
                outputs = self.model(features)

                # Evaluate performance of the model
                self.evaluation(outputs, labels)

        return self.evaluation.get_results(only_last=False)
    
    