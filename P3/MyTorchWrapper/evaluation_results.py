import re
import numpy as np
from typing import Set


class EvaluationResults:
    """Class used to store and retrieve the results of a training or testing process.
    """    
    valid_key = re.compile(r"^[a-z]+(?:_?[a-z]+)*$")

    def __init__(self) -> None:
        self._results = {}
        self._batch_sizes = {}


    def _add_result(self, key: str, result: float, batch_size: int) -> None:
        """Adds a result to the results history.

        Args:
            key (str): Name used to describe the evaluation method (like 'loss' or 'accuracy').
            result (float): The numerical result to store.
            batch_size (int): Size of the batch from which the result has been
            obtained. Used to accurately average the results.
        """        
        assert re.match(self.valid_key, key), "The key can only contain lowercase letters and underscores."
        if key in self._results:
            self._results[key].append(result)
            self._batch_sizes[key].append(batch_size)
        else:
            self._results[key] = [result]
            self._batch_sizes[key] = [batch_size]


    @property
    def methods(self) -> Set[str]:
        """
        Retrieves all the evaluation methods from which there are results available.
        """
        return set(self._results.keys())

    
    def averaged(self) -> 'EvaluationResults':
        new_results = {}
        new_batch_sizes = {}
        for key in self._results.keys():
            a = np.sum(np.multiply(self._results[key], self._batch_sizes[key]))
            b = np.sum(self._batch_sizes[key])
            new_results[key] = [a / b]
            new_batch_sizes[key] = [b]
            
        output = EvaluationResults()
        output._results = new_results
        output._batch_sizes = new_batch_sizes
        return output
    

    def __getitem__(self, key: str):
        """Retrieve the result of a given evaluation method.

        Args:
            key: Name of the evaluation method from which we want to get the results.

        Raises:
            AttributeError: If no results exist for the evaluation method identified by `key`.

        Returns:
            The history of results of the evaluation method identified by `key`.
        """        
        if key in self._results:
            return self._results[key] if len(self._results[key]) > 1 else self._results[key][0]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")


    def __getattr__(self, key: str):
        """Retrieve the result of a given evaluation method as an instance
        attribute `results.loss` or `results.accuracy` for example.

        Args:
            key: Name of the evaluation method from which we want to get the results.

        Raises:
            AttributeError: If no results exist for the evaluation method identified by `key`.

        Returns:
            The history of results of the evaluation method identified by `key`.
        """        
        return self[key]
    

    def __iter__(self):
        for evaluation_method in self.methods:
            yield evaluation_method, self[evaluation_method]
    

    def __str__(self) -> str:
        """Return a string representation of the results.
        """   
        dict = {}
        for m in self.methods:
            dict[m] = self[m]
        return str(dict)
        