import os
from typing import List

from crafting.elements import Item, Zone
from crafting.env import CraftingEnv
from crafting.task import GetItemTask
from crafting.transformation import Transformation
from crafting.world import world_from_transformations


class MiniCraftKeyCorridor(CraftingEnv):
    """Reproduces the KeyCorridor minigrid environment as a crafting environment."""

    START = Zone("start_room")
    """Start room."""
    KEY_ROOM = Zone("key_room")
    """Room containing a key."""
    LOCKED_ROOM = Zone("locked_room")
    """Room behind a locked door."""

    KEY = Item("key")
    """Key used to unlock locked door."""
    BALL = Item("ball")
    """Ball to pickup."""
    WEIGHT = Item("weight")
    """Weight of carried items."""

    OPEN_DOOR = Item("open_door")
    """Opened lockedroom door."""
    OPEN_KEY_DOOR = Item("blue_open_door")
    """Opened keyroom door."""
    CLOSED_KEY_DOOR = Item("blue_closed_door")
    """Closed keyroom door."""
    LOCKED_DOOR = Item("locked_door")
    """Locked door, can be unlocked with a key."""

    def __init__(self, **kwargs) -> None:
        """
        Kwargs:
            See `crafting.env.CraftingEnv`
        """
        resources_path = os.path.join(os.path.dirname(__file__), "resources")
        transformations = self._build_transformations()
        world = world_from_transformations(
            transformations=transformations,
            start_zone=self.START,
            start_zones_items={
                self.LOCKED_ROOM: [self.BALL],
                self.START: [self.LOCKED_DOOR, self.CLOSED_KEY_DOOR],
            },
        )
        self.task = GetItemTask(self.BALL)
        world.resources_path = resources_path
        super().__init__(
            world, purpose=self.task, name="MiniCraftKeyCorridor", **kwargs
        )

    def _build_transformations(self) -> List[Transformation]:
        transformations = []

        search_for_key = Transformation(
            inventory_changes={
                "current_zone": {"add": [self.KEY]},
                "player": {"max": [self.KEY]},
                self.START: {"max": [self.KEY]},
                self.KEY_ROOM: {"max": [self.KEY]},
                self.LOCKED_ROOM: {"max": [self.KEY]},
            },
            zones=[self.KEY_ROOM],
        )
        transformations.append(search_for_key)

        pickup = Transformation(
            inventory_changes={
                "player": {"add": [self.KEY, self.WEIGHT], "max": [self.WEIGHT]},
                "current_zone": {"remove": [self.KEY]},
            },
        )
        put_down = Transformation(
            inventory_changes={
                "player": {"remove": [self.KEY, self.WEIGHT]},
                "current_zone": {"add": [self.KEY]},
            },
        )
        transformations += [pickup, put_down]

        open_door = Transformation(
            inventory_changes={
                "current_zone": {
                    "remove": [self.CLOSED_KEY_DOOR],
                    "add": [self.OPEN_KEY_DOOR],
                },
            },
        )
        transformations.append(open_door)

        move_to_key_room = Transformation(
            destination=self.KEY_ROOM,
            inventory_changes={
                "current_zone": {
                    "remove": [self.OPEN_KEY_DOOR],
                    "add": [self.OPEN_KEY_DOOR],
                },
            },
            zones=[self.START],
        )
        transformations.append(move_to_key_room)

        move_to_start_room = Transformation(
            destination=self.START,
            zones=[self.KEY_ROOM, self.LOCKED_ROOM],
        )
        transformations.append(move_to_start_room)

        unlock_door = Transformation(
            inventory_changes={
                "player": {
                    "remove": [self.KEY],
                    "add": [self.KEY],
                },
                "current_zone": {
                    "remove": [self.LOCKED_DOOR],
                    "add": [self.OPEN_DOOR],
                },
            },
        )
        transformations.append(unlock_door)

        move_to_locked_room = Transformation(
            destination=self.LOCKED_ROOM,
            inventory_changes={
                "current_zone": {
                    "remove": [self.OPEN_DOOR],
                    "add": [self.OPEN_DOOR],
                },
            },
            zones=[self.START],
        )
        transformations.append(move_to_locked_room)

        pickup_ball = Transformation(
            inventory_changes={
                "player": {"add": [self.BALL, self.WEIGHT], "max": [self.WEIGHT]},
                "current_zone": {"remove": [self.BALL]},
            }
        )
        transformations.append(pickup_ball)

        return transformations
