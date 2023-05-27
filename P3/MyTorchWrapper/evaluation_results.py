import re
from typing import Dict, List


class EvaluationResults:
    """Class used to store and retrieve the results of a training or testing process.
    """    
    valid_key = re.compile(r"^[a-z]+(?:_?[a-z]+)*$")

    def __init__(self) -> None:
        self.results = {}


    def _add_result(self, key: str, result: float) -> None:
        """Adds a result to the results history.

        Args:
            key (str): Name used to describe the evaluation method (like 'loss' or 'accuracy').
            result (float): The numerical result to store.
        """        
        assert re.match(self.valid_key, key), "The key can only contain lowercase letters and underscores."
        if key in self.results:
            self.results[key].append(result)
        else:
            self.results[key] = [result]


    def __str__(self) -> str:
        """Get a string representation of the results.

        Returns:
            str: String representation of the results.
        """        
        return str(self.results) # TODO: Most likely there is a better text representation
    

    def __getitem__(self, index_or_slice) -> Dict[str, List[float]]:
        new_dict = {}
        for key in self.results.keys():
            new_dict[key] = self.results[key][index_or_slice]
        return new_dict


    def __getattr__(self, key):
        """Retrieve the result of a given evaluation method as an instance attribute.

        Args:
            key: Name of the evaluation method from which we want to get the results

        Raises:
            AttributeError: If no results exist for the evaluation method identified by `key`.

        Returns:
            The history of results of the evaluation method identified by `key`.
            If the history only contains one element, it returns a float number,
            otherwise it returns a list of floats.
        """        
        if key in self.results:
            return self.results[key] if len(self.results[key]) > 1 else self.results[key][0]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")