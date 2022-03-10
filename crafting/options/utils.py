# Crafting a gym-environment to simultate inventory managment
# Copyright (C) 2021-2022 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Module for utility functions to apply on handcrafted Option. """

from typing import Set

from option_graph import OptionGraph

from crafting.world.items import Item
from crafting.options.actions import SearchItem, CraftRecipe
from crafting.options.options import GetItem


def get_items_in_graph(graph: OptionGraph) -> Set[Item]:
    """Get items in a Crafting option graph.

    Args:
        graph (OptionGraph): An option graph of the Crafting environment.

    Returns:
        Set[Item]: Set of items that appears in the given graph.
    """
    items_in_graph = set()
    for node in graph.nodes():
        if isinstance(node, (SearchItem, GetItem)):
            items_in_graph.add(node.item)
        if isinstance(node, CraftRecipe):
            for itemstack in node.recipe.outputs:
                items_in_graph.add(itemstack.item)
    return items_in_graph
