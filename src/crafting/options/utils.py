# Crafting a gym-environment to simultate inventory managment
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Module for utility functions to apply on handcrafted Option. """

from typing import Dict, Set, Union

from hebg import HEBGraph, Behavior

from crafting.options.actions import CraftRecipe, SearchItem
from crafting.options.options import GetItem, ReachZone
from crafting.world.items import Item


def get_items_in_graph(
    graph: HEBGraph,
    all_options: Dict[str, Union[GetItem, ReachZone]] = None,
) -> Set[Item]:
    """Get items in a Crafting option graph.

    Args:
        graph (HEBGraph): An option graph of the Crafting environment.

    Returns:
        Set[Item]: Set of items that appears in the given graph.
    """
    all_options = all_options if all_options is not None else {}
    items_in_graph = set()
    for node in graph.nodes():
        if isinstance(node, Behavior) and str(node) in all_options:
            node = all_options[str(node)]
        if isinstance(node, (SearchItem, GetItem)):
            if isinstance(node.item, Item):
                items_in_graph.add(node.item)
        if isinstance(node, CraftRecipe):
            if node.recipe.outputs is not None:
                for itemstack in node.recipe.outputs:
                    items_in_graph.add(itemstack.item)
    return items_in_graph
