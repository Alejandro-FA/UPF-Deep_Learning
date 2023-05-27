import re
from os import listdir
import torch
from torch import nn
from typing import Optional


class _PathManager:
    """Auxiliary class in charge of resolving the path of input and output files.
    """    

    model_ext = ".ckpt" # Extension of model files
    summary_ext = ".txt" # Extension of summary files
    filename_pattern = re.compile("model_(\d+).ckpt")

    def __init__(self, models_dir: str) -> None:
        """
        Args:
            models_dir (str): Folder path where the models are stored.
        """                
        self.dir_path = models_dir


    def get_model_name(self, model_id: int) -> str:
        """Given a model id, it returns the model name.
        """        
        return "model_" + str(model_id)
    

    def get_model_path(self, model_id: int) -> str:
        """Given a model id, it returns the path of its corresponding model file (.ckpt)
        """        
        return self.dir_path + self.get_model_name(model_id) + self.model_ext
    

    def get_summary_path(self, model_id: int) -> str:
        """Given a model id, it returns its corresponding training summary file path.
        """        
        return self.dir_path + self.get_model_name(model_id) + self.summary_ext


    def next_id_available(self) -> int:
        """Computes the next index available for storing a new model
        Returns:
            int: next index available
        """        
        files = listdir(self.dir_path)
        models = list(filter(lambda name: _PathManager.model_ext in name, files))
        indices = [int(_PathManager.filename_pattern.search(model).group(1)) for model in models]
        return 0 if not indices else max(indices)



class IOManager:
    """Saves and loads PyTorch models for future reference and use.
    """

    def __init__(self, storage_dir: str) -> None:
        """
        Args:
            storage_dir (str): Folder path where the models are stored.
        """        
        self._path_manager = _PathManager(storage_dir)


    def next_id_available(self) -> int:
        """Returns the next identification number available for a model.
        """        
        return self._path_manager.next_id_available()


    def save(self, model: nn.Module, model_id: int) -> None:
        """Given a torch model, it saves it in the storage_dir with the
        provided model_id.

        Args:
            model (nn.Module): Neural Network model to store
            model_id (int): Identification number with which to store the model.
        """        
        file_path = self._path_manager.get_model_path(model_id)
        torch.save(model.state_dict(), file_path)


    def load(self, model: nn.Module, model_id: int) -> None:
        """Given a torch model and a model_id, it loads all the parameters stored
        in the model file (identified with model_id) inside the model.

        Args:
            model (nn.Module): Neural Network model in which sto store the paramers. It must have the appropriate architecture.
            model_id (int): Identification number of the model to load.
        """        
        file_path = self._path_manager.get_model_path(model_id)
        model.load_state_dict(torch.load(file_path))


    def save_summary(self, summary_content: str, model_id: int) -> None:
        """Given a training summary, it stores its results in a file.
        Args:
            summary_content (str): The content of the summary.
            model_id (int): Identification number of the model from which the
            the summary has been obtained.
        """        
        file_path = self._path_manager.get_summary_path(model_id)
        with open(file_path, "w") as results_txt:
            results_txt.write(summary_content)
    