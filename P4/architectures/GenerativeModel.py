"""Defines common behavior for all Generative Models.

This module defines an interface that all Generative Models should follow.
"""
import torch
from abc import ABC, abstractmethod


class GenerativeModel(ABC):
    """Abstract Base Class for Generative Models.
    
    All Generative Models should inherit from this class. You can change it to
    add new functionality. You can also modify the current interface, but you
    should be careful of not breaking the code. 
    """
    def sample(self, n_samples: int, device:torch.device=torch.device('cpu')) -> torch.Tensor:
        """Sample a set of generated images (from random vectors z of the
        latent space).

        Args:
            n_samples (int): Number of images that you wish to generate.
            device (torch.device, optional): Torch device to use. Defaults to 'cpu'.

        Returns:
            torch.Tensor: Output images as a Tensor.
        """        
        assert(n_samples >= 1)
        z = self.get_latent_space(n_samples, device)
        return self.decode(z)


    @abstractmethod
    def get_latent_space(self, n_samples: int, device:torch.device=torch.device('cpu')) -> torch.Tensor:
        """Get a random vector from the latent space.

        Args:
            n_samples (int): Number of random vectors desired (you need one
            for each image that you wish to generate).
            device (str, optional): Torch device to use. Defaults to 'cpu'.
        """
        
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode an image from latent space to pixels.

        Args:
            z (torch.Tensor): Latent space vector.

        Returns:
            torch.Tensor: Image(s) as an array of pixels.
        """        


    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the concrete name of your GenerativeModel. Used for writing
        output files.

        Returns:
            str: The name of your model.
        """        