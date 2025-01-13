from torchrl.envs import EnvBase
from tensordict.tensordict import TensorDict
import torch

from .gamestate import CatanGameState

class CatanEnvironment(EnvBase):
    def __init__(self):
        super().__init__()
        self.observation_space = {"state": torch.zeros(CatanGameState.get_observation_space_dim())}
        self.action_space = {"action": torch.zeros(2)}      # Example: 2 possible actions
        self.state = torch.zeros(4)                         # Current state
        self.done = False                                   # Termination flag