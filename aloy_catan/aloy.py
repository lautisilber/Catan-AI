from torch import multiprocessing
import torch
from tqdm import tqdm

from .gamestate import CatanGameState


class AloyCatan:
    def __init__(self) -> None:
        is_fork = multiprocessing.get_start_method() == "fork"
        self.device = (
            torch.device('mps') if torch.backends.mps.is_available() else
            torch.device(0) if torch.cuda.is_available() and not is_fork else
            torch.device('cpu')
        )

        # PPO parameters
        sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
        num_epochs = 10  # optimization steps per batch of data collected
        clip_epsilon = (
            0.2  # clip value for PPO loss: see the equation in the intro for more context.
        )
        gamma = 0.99
        lmbda = 0.95
        entropy_eps = 1e-4

        

def create_aloy_catan() -> AloyCatan:
    pass