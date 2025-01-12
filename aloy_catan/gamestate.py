from dataclasses import dataclass
from typing import ClassVar
from pycatan import board
import pycatan
import torch
import numpy as np

@dataclass
class CatanGameState:
    # We're ignoring development cards for now
    # We're only playing 1v1 for now
    self_player: pycatan.Player
    other_player: pycatan.Player

    tiles: set[board.Hex]
    harbors: set[board.Harbor]

    robber: board.Coords
    vertices: tuple[board.Intersection]
    edges: tuple[board.Path]

    dice_history: list[int]

    turn: int
    setup_round: int

    N_TILES: ClassVar[int] = 19
    TILE_ENC_SIZE: ClassVar[int] = 6

    N_HARBORS: ClassVar[int] = 9
    HARBOR_ENC_SIZE: ClassVar[int] = 3

    ROBBER_ENC_SIZE: ClassVar[int] = 2

    N_VERTICES: ClassVar[int] = 52
    VERTEX_ENC_SIZE: ClassVar[int] = 4

    N_EDGES: ClassVar[int] = 70
    EDGE_ENC_SIZE: ClassVar[int] = 6

    OWN_RESOURCES_ENC_SIZE: ClassVar[int] = 5
    OTHER_RESOURCES_ENC_SIZE: ClassVar[int] = 1

    TURN_ENC_SIZE: ClassVar[int] = 1
    SETUP_ROUND_ENC_SIZE: ClassVar[int] = 1

    def __post_init__(self) -> None:
        assert len(self.tiles) == CatanGameState.N_TILES
        assert len(self.harbors) == CatanGameState.N_HARBORS
        assert len(self.vertices) == CatanGameState.N_VERTICES
        assert len(self.edges) == CatanGameState.N_EDGES
        assert self.turn >= 0

    @staticmethod
    def encode_coords(c: board.Coords) -> tuple[int, int]:
        return (c.q, c.r)
    
    @staticmethod
    def encode_tile_type(rt: board.HexType) -> int:
        return {
            board.HexType.FOREST: 0,
            board.HexType.HILLS: 1,
            board.HexType.PASTURE: 2,
            board.HexType.FIELDS: 3,
            board.HexType.MOUNTAINS: 4,
            board.HexType.DESERT: 5,
        }[rt]

    @staticmethod
    def encode_resource_type(rt: pycatan.Resource) -> int:
        return {
            pycatan.Resource.LUMBER: 0,
            pycatan.Resource.BRICK: 1,
            pycatan.Resource.WOOL: 2,
            pycatan.Resource.GRAIN: 3,
            pycatan.Resource.ORE: 4,
        }[rt]
    
    @staticmethod
    def encode_player(player: pycatan.Player, self_player: pycatan.Player) -> int:
        return 1 - (player is self_player) # 0 is self, 1 is other
    
    @staticmethod
    def encode_building_vertex(bt: board.IntersectionBuilding | None, self_player: pycatan.Player) -> tuple[int, int]:
        # return (building type, owner)}
        if (bt is None):
            return (0, 0)
        type = {
            board.BuildingType.SETTLEMENT: 1,
            board.BuildingType.CITY: 2,
        }[bt.building_type]
        owner = CatanGameState.encode_player(bt.owner, self_player)
        return (type, owner)
    
    @staticmethod
    def encode_building_edge(bt: board.PathBuilding | None, self_player: pycatan.Player) -> tuple[int, int]:
        # return (building type, owner)}
        if (bt is None):
            return (0, 0)
        type = {
            board.BuildingType.ROAD: 1,
        }[bt.building_type]
        owner = CatanGameState.encode_player(bt.owner, self_player)
        return (type, owner)
    
    @staticmethod
    def encode_tile(tile: board.Hex) -> tuple[int, int, int, int]:
        coords = CatanGameState.encode_coords(tile.coords)
        hex_type = CatanGameState.encode_tile_type(tile.hex_type)
        token = tile.token_number if tile.token_number is not None else 0
        return (*coords, hex_type, token)
    
    @staticmethod
    def encode_harbor(harbor: board.Harbor) -> tuple[int, int, int, int, int]:
        # return (coord1 q, coord1 r, coord2 q, coord2 r, harbor type)
        type = 5 if not isinstance(harbor.resource, pycatan.Resource) else CatanGameState.encode_resource_type(harbor.resource)
        coords = list(harbor.path_coords)
        return (*CatanGameState.encode_coords(coords[0]), *CatanGameState.encode_coords(coords[1]), type)
    
    @staticmethod
    def encode_robber(robber: board.Coords) -> tuple[int, int]:
        return CatanGameState.encode_coords(robber)
    
    @staticmethod
    def encode_vertex(vert: board.Intersection, self_player: pycatan.Player) -> tuple[int, int, int, int]:
        coords = CatanGameState.encode_coords(vert.coords)
        building_type_and_owner = CatanGameState.encode_building_vertex(vert.building, self_player)
        return (*coords, *building_type_and_owner)
    
    @staticmethod
    def encode_edge(edge: board.Path, self_player: pycatan.Player) -> tuple[int, int, int, int, int, int]:
        coords = list(edge.path_coords)
        building_type_and_owner = CatanGameState.encode_building_edge(edge.building, self_player)
        return (*CatanGameState.encode_coords(coords[0]), *CatanGameState.encode_coords(coords[1]), *building_type_and_owner)
    
    @staticmethod
    def encode_own_resources(self_player: pycatan.Player) -> tuple[int, int, int, int, int]:
        return (
            self_player.resources[pycatan.Resource.LUMBER],
            self_player.resources[pycatan.Resource.BRICK],
            self_player.resources[pycatan.Resource.WOOL],
            self_player.resources[pycatan.Resource.GRAIN],
            self_player.resources[pycatan.Resource.ORE],
        )

    @staticmethod
    def encode_other_resources(other_player: pycatan.Player) -> int:
        return len(other_player.resources)

    def to_tensor(self, dice_history_len: int) -> np.ndarray:
        '''
        - 19 tiles
            - 2 for coords
            - 1 for resource type
            - 1 for number
        - 9 harbors
            - 4 for coords x2
            - 1 for type
        - 1 robber
            - 2 for coords
        - 52 vertices
            - 2 coords
            - 1 building type
            - 1 player on the vertex
        - 70 edges
            - 4 coords x2
            - 1 populated with road
            - 1 player on the edge
        - <dice_history_len> dice_history
            - 1 dice roll
        - own_resources
            - 5 resource types (with an int indicating how many of them you have)
        - other_player_resources
        - 1 turn (turn number)
        '''

        # encoded_state = np.zeros(n_tiles * tile_enc_size + n_harbors * harbor_enc_size + robber_enc_size + n_vertices * vertex_enc_size + n_edges * edge_enc_size + dice_history_len + own_res_enc_size + other_res_enc_size + 1)
        encoded_state = np.zeros(
            CatanGameState.N_TILES * CatanGameState.TILE_ENC_SIZE +
            CatanGameState.N_HARBORS * CatanGameState.HARBOR_ENC_SIZE +
            CatanGameState.ROBBER_ENC_SIZE +
            CatanGameState.N_VERTICES * CatanGameState.VERTEX_ENC_SIZE +
            CatanGameState.N_EDGES * CatanGameState.EDGE_ENC_SIZE +
            CatanGameState.OWN_RESOURCES_ENC_SIZE +
            CatanGameState.OTHER_RESOURCES_ENC_SIZE +
            CatanGameState.TURN_ENC_SIZE +
            CatanGameState.SETUP_ROUND_ENC_SIZE
        )

        offset = 0
        # tiles
        for tile in self.tiles:
            encoded_state[offset:offset + CatanGameState.TILE_ENC_SIZE] = CatanGameState.encode_tile(tile)
            offset += CatanGameState.TILE_ENC_SIZE
        # harbors
        for harbor in self.harbors:
            encoded_state[offset:offset + CatanGameState.HARBOR_ENC_SIZE] = CatanGameState.encode_harbor(harbor)
            offset += CatanGameState.HARBOR_ENC_SIZE
        # robber
        encoded_state[offset:offset + CatanGameState.ROBBER_ENC_SIZE] = CatanGameState.encode_robber(self.robber)
        offset += CatanGameState.ROBBER_ENC_SIZE
        # vertices
        for vertex in self.vertices:
            encoded_state[offset:offset + CatanGameState.VERTEX_ENC_SIZE] = CatanGameState.encode_vertex(vertex, self.self_player)
            offset += CatanGameState.VERTEX_ENC_SIZE
        # edges
        for edge in self.edges:
            encoded_state[offset:offset + CatanGameState.EDGE_ENC_SIZE] = CatanGameState.encode_edge(edge, self.self_player)
            offset += CatanGameState.EDGE_ENC_SIZE
        # dice history
        encoded_state[offset:offset + dice_history_len] = self.dice_history
        offset += dice_history_len
        # own resources
        encoded_state[offset:offset + CatanGameState.OWN_RESOURCES_ENC_SIZE] = CatanGameState.encode_own_resources(self.self_player)
        offset += CatanGameState.OWN_RESOURCES_ENC_SIZE
        # other player resources
        encoded_state[offset:offset + CatanGameState.OTHER_RESOURCES_ENC_SIZE] = CatanGameState.encode_other_resources(self.other_player)
        offset += CatanGameState.OTHER_RESOURCES_ENC_SIZE
        # turn
        encoded_state[offset:offset + CatanGameState.TURN_ENC_SIZE] = self.turn
        offset += CatanGameState.TURN_ENC_SIZE
        # setup round
        encoded_state[offset:offset + CatanGameState.SETUP_ROUND_ENC_SIZE] = self.setup_round
        offset += CatanGameState.SETUP_ROUND_ENC_SIZE

        return encoded_state

