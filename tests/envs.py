from crafting.elements import Item, ItemStack, Zone
from crafting.env import CraftingEnv
from crafting.transformation import Transformation
from crafting.world import world_from_transformations


def classic_env():
    start_zone = Zone("start")
    other_zone = Zone("other_zone")
    zones = [start_zone, other_zone]

    move_to_other_zone = Transformation(
        destination=other_zone,
        zones=[start_zone],
    )

    wood = Item("wood")
    search_wood = Transformation(
        inventory_changes={
            "player": {"add": [wood]},
        }
    )

    stone = Item("stone")
    search_stone = Transformation(
        inventory_changes={
            "player": {"add": [ItemStack(stone, 1)]},
        }
    )

    plank = Item("plank")
    craft_plank = Transformation(
        inventory_changes={
            "player": {"remove": [wood], "add": [ItemStack(plank, 4)]},
        }
    )

    table = Item("table")
    craft_table = Transformation(
        inventory_changes={
            "player": {"remove": [ItemStack(plank, 4)]},
            "current_zone": {"add": [ItemStack(table)]},
        }
    )

    wood_house = Item("wood house")
    build_house = Transformation(
        inventory_changes={
            "player": {"remove": [ItemStack(plank, 32), ItemStack(wood, 8)]},
            "current_zone": {"add": [ItemStack(wood_house)]},
        }
    )

    items = [wood, stone, plank]
    zones_items = [table, wood_house]
    named_transformations = {
        "move_to_other_zone": move_to_other_zone,
        "search_wood": search_wood,
        "search_stone": search_stone,
        "craft_plank": craft_plank,
        "craft_table": craft_table,
        "build_house": build_house,
    }

    world = world_from_transformations(
        transformations=list(named_transformations.values()),
        start_zone=start_zone,
    )
    env = CraftingEnv(world)
    return env, world, named_transformations, start_zone, items, zones, zones_items


def player_only_env():
    wood = Item("wood")
    search_wood = Transformation(
        inventory_changes={
            "player": {"add": [wood]},
        }
    )

    stone = Item("stone")
    search_stone = Transformation(
        inventory_changes={
            "player": {"add": [ItemStack(stone, 1)]},
        }
    )

    plank = Item("plank")
    craft_plank = Transformation(
        inventory_changes={
            "player": {"remove": [wood], "add": [ItemStack(plank, 4)]},
        }
    )

    items = [wood, stone, plank]
    named_transformations = {
        "search_wood": search_wood,
        "search_stone": search_stone,
        "craft_plank": craft_plank,
    }

    world = world_from_transformations(
        transformations=list(named_transformations.values())
    )
    env = CraftingEnv(world)
    return env, world, named_transformations, None, items, [], []


def zone_only_env():
    start_zone = Zone("start")
    wood = Item("wood")
    wood = Item("wood")
    search_wood = Transformation(
        inventory_changes={
            "current_zone": {"add": [wood]},
        }
    )

    stone = Item("stone")
    search_stone = Transformation(
        inventory_changes={
            "current_zone": {"add": [ItemStack(stone, 1)]},
        }
    )

    plank = Item("plank")
    craft_plank = Transformation(
        inventory_changes={
            "current_zone": {"remove": [wood], "add": [ItemStack(plank, 4)]},
        }
    )

    zones_items = [wood, stone, plank]
    named_transformations = {
        "search_wood": search_wood,
        "search_stone": search_stone,
        "craft_plank": craft_plank,
    }

    world = world_from_transformations(
        transformations=list(named_transformations.values()),
        start_zone=start_zone,
    )
    env = CraftingEnv(world)
    return env, world, named_transformations, start_zone, [], [], zones_items
