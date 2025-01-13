from dataclasses import dataclass
import pycatan
from pycatan import board
from typing import ClassVar
import numpy as np

# No robber for now

@dataclass
class CatanGameAction_BuildSettlement:
    enc_size: ClassVar[int] = 1 + 2 # one bit to check if this action should be made, two bits for coords

    player: pycatan.Player
    coords: board.Coords
    # Whether to remove the resources required to build a
    # settlement from the player's hands, and raise an error
    # if they don't have them
    cost_resources: bool = True
    # Whether to raise an error if the settlement would
    # not be connected to a road owned by the same player.
    ensure_connected: bool = True

@dataclass
class CatanGameAction_BuildRoad:
    enc_size: ClassVar[int] = 1 + 2

    player: pycatan.Player
    coords: set[board.Coords]
    # Whether to remove the resources required to build a
    # settlement from the player's hands, and raise an error
    # if they don't have them
    cost_resources: bool = True

@dataclass
class CatanGameAction_UpgradeSettlementToCity:
    enc_size: ClassVar[int] = 1 + 2
    
    player: pycatan.Player
    coords: set[board.Coords]
    # Whether to remove the resources required to build a
    # settlement from the player's hands, and raise an error
    # if they don't have them
    cost_resources: bool = True
    # Whether to raise an error if the settlement would
    # not be connected to a road owned by the same player.
    ensure_connected: bool = True


class CatanGameAction:
    def __init__(self, action_arr: np.ndarray) -> None:
        # actions will be encoded hte following way
        # the first int is a range [0, 1]. If > 0.5, the action is taken
        # the other two ints are also in range [0, 1] and correspond to
        # the r and q coordinates. The range will be mapped to the board's
        # range and approximated to the nearest coord
        self.action_arr = action_arr
