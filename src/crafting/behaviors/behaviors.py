# Crafting a gym-environment to simultate inventory managment
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Module for handcrafted Behavior with HEBGraph in any Crafting environment. """

from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np
from hebg import HEBGraph, Behavior

from crafting.behaviors.actions import CraftRecipe, MoveToZone, SearchItem
from crafting.behaviors.feature_conditions import HasItem, HasProperty, IsInZone
from crafting.render.utils import load_or_create_image

if TYPE_CHECKING:
    from crafting.world.items import Item
    from crafting.world.world import World
    from crafting.world.zones import Zone


class ReachZone(Behavior):

    """Behavior for moving to a Zone"""

    def __init__(self, zone: "Zone", world: "World"):
        super().__init__(f"Reach {str(zone)}")
        self.world = world
        self.zone = zone

    def build_graph(self) -> HEBGraph:
        """Build the HEBGraph of this Behavior.

        Returns:
            The built HEBGraph.

        """
        graph = HEBGraph(behavior=self)
        go_to_zone = MoveToZone(self.zone, self.world)
        graph.add_node(go_to_zone)
        return graph


class GetItem(Behavior):

    """Behavior for getting an item"""

    def __init__(
        self,
        world: "World",
        item: "Item",
        all_behaviors: Dict[Union[int, str], Behavior],
        items_needed: List[List[tuple]],
        last_action: tuple,
        zones_id_needed: list = None,
        zones_properties_needed: dict = None,
    ):

        super().__init__(name=f"Get {str(item)}")
        self.world = world
        self.item = item

        self.items_needed = items_needed
        if self.items_needed is None:
            self.items_needed = [[]]

        self.zones_id_needed = zones_id_needed
        if self.zones_id_needed is None:
            self.zones_id_needed = []

        self.zones_properties_needed = zones_properties_needed
        if self.zones_properties_needed is None:
            self.zones_properties_needed = {}

        self.last_action = last_action
        self.all_behaviors = all_behaviors

    def build_graph(self) -> HEBGraph:
        """Build the HEBGraph of this Behavior.

        Returns:
            The built HEBGraph.

        """
        graph = HEBGraph(behavior=self, all_behaviors=self.all_behaviors)
        prev_checks = []

        # Any of Craft behaviors
        if None not in self.items_needed:
            for craft_behavior in self.items_needed:
                prev_check = None
                for item_id, quantity in craft_behavior:
                    has_item = self._add_crafting_behavior(graph, item_id, quantity)
                    if prev_check is not None:
                        graph.add_edge(prev_check, has_item, index=int(True))
                    prev_check = has_item
                if prev_check is not None:
                    prev_checks.append(prev_check)

        # Any of the zones possibles
        prev_checks_zone = []
        if self.world.n_zones > 1:
            for zone_id in self.zones_id_needed:
                is_in_zone = self._add_zone_behavior(graph, zone_id)
                prev_checks_zone.append(is_in_zone)
                for prev in prev_checks:
                    graph.add_edge(prev, is_in_zone, index=int(True))
        if len(prev_checks_zone) > 0:
            prev_checks = prev_checks_zone

        # All properties needed
        for prop, _ in self.zones_properties_needed.items():
            has_prop = self._add_property_needed(graph, prop)
            for prev in prev_checks:
                graph.add_edge(prev, has_prop, index=int(True))
            prev_checks = [has_prop]

        # Add last action
        action_type, obj_id = self.last_action
        if action_type == "get":
            item = self.world.item_from_id[obj_id]
            action = SearchItem(item, self.world)
        elif action_type == "craft":
            recipe = self.world.recipes_from_id[obj_id]
            action = CraftRecipe(recipe, self.world)
        elif action_type == "move":
            zone = self.world.zone_from_id[obj_id]
            action = MoveToZone(zone, self.world)
        else:
            raise ValueError(f"Unknowed action_type: {action_type}")

        graph.add_node(action)
        for prev in prev_checks:
            graph.add_edge(prev, action, index=int(True))
        return graph

    def _add_crafting_behavior(
        self, graph: HEBGraph, item_id: int, quantity: int
    ) -> HasItem:
        item = self.world.item_from_id[item_id]
        has_item = HasItem(item=item, world=self.world, quantity=quantity)
        graph.add_node(has_item)
        image = np.array(load_or_create_image(self.world, item))
        get_item = Behavior(f"Get {item}", image=image)
        graph.add_node(get_item)
        graph.add_edge(has_item, get_item, index=int(False))
        return has_item

    def _add_zone_behavior(self, graph: HEBGraph, zone_id: int) -> IsInZone:
        zone = self.world.zone_from_id[zone_id]
        is_in_zone = IsInZone(zone, self.world)
        graph.add_node(is_in_zone)
        image = np.array(load_or_create_image(self.world, zone))
        reach_zone = Behavior(f"Reach {zone}", image=image)
        graph.add_node(reach_zone)
        graph.add_edge(is_in_zone, reach_zone, index=int(False))
        return is_in_zone

    def _add_property_needed(self, graph: HEBGraph, prop: str) -> HasProperty:
        has_prop = HasProperty(prop, world=self.world)
        graph.add_node(has_prop)
        image = np.array(load_or_create_image(self.world, prop))
        get_prop = Behavior(f"Get {prop}", image=image)
        graph.add_node(get_prop)
        graph.add_edge(has_prop, get_prop, index=int(False))
        return has_prop