import torchinfo
from .train import Trainer
from .evaluation_results import EvaluationResults


def training_summary(trainer: Trainer, test_results: EvaluationResults) -> str:    
    """Build a performance summary report for future reference.

    Args:
        trainer (Trainer): trainer instance used to train the model.
        test_results (EvaluationResults): EvaluationResults obtained during
        testing. Used to record the performance of the model.

    Returns:
        str: Summary content. Usually used for printing it to screen or
        writing it to a file.
    """
    batch, _ = next(iter(trainer.data_loader))
    model_stats = torchinfo.summary(trainer.model, input_size=batch.shape, device=trainer.device, verbose=0)

    summary = (
        f"Test results: {test_results}\n"
        + f"Loss function used: {trainer.evaluation.loss_criterion}\n"
        + f"Epochs: {trainer.epochs}\n"
        + f"Optimizer: {trainer.optimizer}\n"
        + str(model_stats)
    )

    return summary