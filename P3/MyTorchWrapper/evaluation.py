import torch
from .evaluation_results import EvaluationResults


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


    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor, results: EvaluationResults) -> torch.Tensor:
        """Evaluates the performance of the output of a torch model and returns
        the loss.

        Args:
            outputs (torch.Tensor): the output of the model
            labels (torch.Tensor): the target labels of each point / sample
            results (EvaluationResults): where to store the results
        """        
        loss = self.loss_criterion(outputs, labels)
        results._add_result('loss', loss.item())
        return loss
    


class AccuracyEvaluation(BasicEvaluation):
    """Besides computing the loss of a model, it also computes the accuracy of
    the output. Only intended for classification models.
    """    

    def __init__(self, loss_criterion) -> None:
        super().__init__(loss_criterion)
        self.accuracy_list = []


    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor, results: EvaluationResults) -> torch.Tensor:
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, dim=1)  # Get predicted class
            total = labels.size(0)
            correct = (predicted == labels).sum().item()  # Compare with ground-truth
            accuracy = 100 * correct / total
            results._add_result('accuracy', accuracy)

        return super().__call__(outputs, labels, results)
    
    