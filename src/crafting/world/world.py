# Crafting a gym-environment to simultate inventory managment
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" A World stiches together every non-player parts of the environment.

"""

import os
from typing import TYPE_CHECKING, Dict, List

import numpy as np

import crafting
from crafting.behaviors.behaviors import GetItem, ReachZone
from crafting.world.items import Tool
from crafting.world.requirement_graph import (
    build_requirements_graph,
    draw_requirements_graph,
)

if TYPE_CHECKING:
    from hebg.behavior import Behavior
    from crafting.world.items import Item
    from crafting.world.recipes import Recipe
    from crafting.world.zones import Zone


class World:

    """A crafting World containing items, recipes and zones."""

    def __init__(
        self,
        items: List["Item"],
        recipes: List["Recipe"],
        zones: List["Zone"],
        searchable_items: List["Item"] = None,
        resources_path: str = None,
        font_path: str = None,
    ):
        """A crafting World containing items, recipes and zones.

        Args:
            items: List of world items.
            recipes: List of world recipes.
            zones: List of world zones.

        """
        # Zones
        self.zones = zones
        self.zone_from_id = {zone.zone_id: zone for zone in zones}
        self.zone_id_to_slot = {zone.zone_id: i for i, zone in enumerate(zones)}
        self.n_zones = len(zones)

        # Items
        self.items = items
        self.item_from_id = {item.item_id: item for item in items}
        self.item_from_name = {item.name: item for item in items}
        self.item_id_to_slot = {item.item_id: i for i, item in enumerate(items)}
        self.n_items = len(items)

        self.tools = [item for item in self.items if isinstance(item, Tool)]

        # Foundable items
        self.foundable_items_ids = set()
        for zone in zones:
            for item in zone.items:
                self.foundable_items_ids.add(item.item_id)

        self.foundable_items = [
            self.item_from_id[item_id] for item_id in self.foundable_items_ids
        ]
        self.foundable_items_id_to_slot = {
            item.item_id: i for i, item in enumerate(self.foundable_items)
        }
        self.n_foundable_items = len(self.foundable_items)

        self.searchable_items = (
            searchable_items if searchable_items is not None else self.foundable_items
        )

        # Recipes
        self.recipes = recipes
        self.recipes_from_id = {recipe.recipe_id: recipe for recipe in recipes}
        self.recipes_id_to_slot = {
            recipe.recipe_id: i for i, recipe in enumerate(recipes)
        }
        self.n_recipes = len(recipes)
        self.craftable_items: List["Item"] = []
        for recipe in recipes:
            if recipe.outputs is not None:
                output_items = [stack.item for stack in recipe.outputs]
                self.craftable_items += output_items

        # Getable items
        self.getable_items: List["Item"] = []
        for item in self.foundable_items:
            if hasattr(item, "items_dropped"):
                self.getable_items += item.items_dropped
            else:
                self.getable_items.append(item)
        self.getable_items += self.craftable_items

        # Zone properties
        self.zone_properties = set()
        for zone in zones:
            for prop in zone.properties:
                self.zone_properties.add(prop)

        for recipe in recipes:
            for prop in recipe.added_properties:
                self.zone_properties.add(prop)

        self.zone_properties: List[str] = np.array(list(self.zone_properties))
        self.property_to_slot = {
            prop: i + self.n_items + self.n_zones
            for i, prop in enumerate(self.zone_properties)
        }
        self.n_zone_properties = len(self.zone_properties)

        self.n_actions = self.n_foundable_items + self.n_recipes + self.n_zones
        self.observation_size = self.n_items + self.n_zones + self.n_zone_properties

        # Resources path
        render_path = os.path.dirname(crafting.render.__file__)
        default_ressources_path = os.path.join(render_path, "default_ressources")

        if resources_path is None:
            self.resources_path = default_ressources_path
        else:
            self.resources_path = resources_path

        if font_path is None:
            self.font_path = os.path.join(default_ressources_path, "font.ttf")
        else:
            self.font_path = font_path

        self._requirements_graph = None

    def action(self, action_type: str, identification: int) -> int:
        """Return the action_id from action_type and identification."""
        action_id = 0
        if action_type == "get":
            action_id += self.foundable_items_id_to_slot[identification]
        elif action_type == "craft":
            action_id = self.n_foundable_items
            action_id += self.recipes_id_to_slot[identification]
        elif action_type == "move":
            action_id = self.n_foundable_items + self.n_recipes
            action_id += self.zone_id_to_slot[identification]
        return action_id

    def action_from_id(self, action_id: int) -> str:
        """Describe the action_id effects."""
        offset = 0
        if action_id < self.n_foundable_items:
            action_type = "get"
            object_concerned = self.foundable_items[action_id]
        elif 0 <= action_id - self.n_foundable_items < self.n_recipes:
            offset = self.n_foundable_items
            action_type = "craft"
            object_concerned = self.recipes[action_id - offset]
        elif action_id >= self.n_foundable_items + self.n_recipes:
            action_type = "move"
            offset = self.n_foundable_items + self.n_recipes
            object_concerned = self.zones[action_id - offset]
        return f"{action_type.capitalize()} {object_concerned}"

    def zone_id_from_observation(self, observation):
        """Return the player zone from an observation."""
        one_hot_zones = observation[self.n_items : self.n_items + self.n_zones]
        zone_slot = np.where(one_hot_zones)[0][0]
        return self.zones[zone_slot].zone_id

    def properties_from_observation(self, observation):
        """Return the zone proprietes from an observation."""
        one_hot_props = observation[self.n_items + self.n_zones :]
        props_slot = np.where(one_hot_props)
        return self.zone_properties[props_slot]

    def get_all_behaviors(self) -> Dict[str, "Behavior"]:
        """Return a dictionary of handcrafted behaviors to get each item, zone and property."""
        all_behaviors = {}

        for zone in self.zones:
            zone_behavior = ReachZone(zone, self)
            all_behaviors[str(zone_behavior)] = zone_behavior

        for item in self.foundable_items:
            zones_id_needed = []
            for zone in self.zones:
                if item.item_id in zone.items:
                    zones_id_needed.append(zone.zone_id)

            items_needed = []
            if item.required_tools is not None:
                for tool in item.required_tools:
                    crafting_behavior = (
                        [(tool.item_id, 1)] if tool is not None else None
                    )
                    items_needed.append(crafting_behavior)

            if hasattr(item, "items_dropped"):
                for dropped_item in item.items_dropped:
                    item_behavior = GetItem(
                        world=self,
                        item=dropped_item,
                        all_behaviors=all_behaviors,
                        items_needed=items_needed,
                        last_action=("get", item.item_id),
                        zones_id_needed=zones_id_needed,
                    )
            else:
                item_behavior = GetItem(
                    world=self,
                    item=item,
                    all_behaviors=all_behaviors,
                    items_needed=items_needed,
                    last_action=("get", item.item_id),
                    zones_id_needed=zones_id_needed,
                )
            all_behaviors[str(item_behavior)] = item_behavior

        for recipe in self.recipes:

            items_needed = [
                [(itemstack.item_id, itemstack.size) for itemstack in recipe.inputs]
            ]

            if recipe.outputs is not None:
                for output in recipe.outputs:
                    recipe_behavior = GetItem(
                        world=self,
                        item=output.item,
                        all_behaviors=all_behaviors,
                        items_needed=items_needed,
                        zones_properties_needed=recipe.needed_properties,
                        last_action=("craft", recipe.recipe_id),
                    )
                    all_behaviors[str(recipe_behavior)] = recipe_behavior

            if recipe.added_properties is not None:
                for zone_property in recipe.added_properties:
                    zone_property_behavior = GetItem(
                        world=self,
                        item=zone_property,
                        all_behaviors=all_behaviors,
                        items_needed=items_needed,
                        zones_properties_needed=recipe.needed_properties,
                        last_action=("craft", recipe.recipe_id),
                    )
                    all_behaviors[str(zone_property_behavior)] = zone_property_behavior

        return all_behaviors

    @property
    def requirements_graph(self):
        """Graph of the requirements for each item, zone or property in this World."""
        if self._requirements_graph is None:
            self._requirements_graph = build_requirements_graph(self)
        return self._requirements_graph

    def draw_requirements_graph(self, ax):
        """Draw the requirements_graph of this world on a given matplotlib.Axes."""
        draw_requirements_graph(ax, self.requirements_graph)

    def __str__(self):
        world_str = "Items: "
        for item in self.items:
            requitements_txt = ""
            if item.required_tools is not None:
                requitements_txt = f"<- {item.required_tools}"
            world_str += f";{item}{requitements_txt}"

        world_str += "| Zones "
        for zone in self.zones:
            world_str += ";" + repr(zone)

        world_str += "| Recipes "
        for recipe in self.recipes:
            world_str += ";" + str(recipe)

        return world_str
