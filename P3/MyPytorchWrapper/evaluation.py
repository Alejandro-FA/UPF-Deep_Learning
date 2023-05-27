import torch
from typing import Dict


class BasicEvaluation:
    """
    Class used to evaluate the performance of a model. Different problems
    require different evaluation methods, so this class attempts to encapsulate
    this behaviour for more flexibility.
    
    It is used by the Trainer class and the Tester class.

    BasicEvaluation only computes the loss to evaluate the performance of a
    mode. Additional evaluation methods can be added by subclassing this class.
    """    

    def __init__(self, loss_criterion) -> None:
        """
        Args:
            loss_criterion (torch.nn.modules.loss): the loss function to use
        """            
        self.loss_criterion = loss_criterion
        self.loss_list = []


    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Evaluates the performance of the output of a torch model and returns
        the loss.

        Args:
            outputs (torch.Tensor): the output of the model
            labels (torch.Tensor): the target labels of each point / sample
        """        
        loss = self.loss_criterion(outputs, labels)
        self.loss_list.append(loss.item())
        return loss


    def get_results(self, only_last: bool = False) -> Dict[str, float]:
        """Gets the results of previous evaluation calls.

        Args:
            only_last (bool, optional): Whether to just get the last result or
            the whole history. Defaults to False.

        Returns:
            Dict[str, float]: A dictionary with the performance results.
        """       
        loss = self.loss_list[-1] if only_last else self.loss_list
        results_dict = {"loss": loss}
        return results_dict


    def copy(self): # FIXME: Add return type `typing.Self`` in python 3.11
        """Create a shallow copy of the current instance (the evaluation
        history is not copied).

        Returns:
            BasicEvaluation: A new evaluation instance with the same configuration.
        """ 
        return self(loss_criterion=self.loss_criterion)
    


class AccuracyEvaluation(BasicEvaluation):
    """Besides computing the loss of a model, it also computes the accuracy of
    the output. Only intended for classification models.
    """    

    def __init__(self, loss_criterion) -> None:
        super().__init__(loss_criterion)
        self.accuracy_list = []


    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, dim=1)  # Get predicted class
            total = labels.size(0)
            correct = (predicted == labels).sum().item()  # Compare with ground-truth
            accuracy = 100 * correct / total
            self.accuracy_list.append(accuracy)

        return super().__call__(outputs, labels)
    

    def get_results(self, only_last: bool = False) -> Dict[str, float]:
        results_dict = super().get_results(only_last)
        accuracy = self.accuracy_list[-1] if only_last else self.accuracy_list
        results_dict['accuracy'] = accuracy
        return results_dict
    