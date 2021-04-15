# Crafting a gym-environment to simultate inventory managment
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from typing import List, Dict, Union
from copy import deepcopy

import networkx as nx

from crafting.world.items import Item
from crafting.world.zones import Zone
from crafting.graph import compute_levels, option_layout

class Option():

    def __call__(self, observations, greedy: bool=False):
        """ Use the option to get next actions.

        Args:
            observations: Observations of the environment.
            greedy: If true, the agent should act greedily.

        Returns:
            actions: Actions given by the option with current observations.
            option_done: True if the option is done, False otherwise.

        """
        raise NotImplementedError


class GoToZone(Option):

    def __init__(self, zone:Zone, world:"crafting.world.World"):
        self.world = world
        self.zone = zone

    def __call__(self, observations, greedy: bool=False):
        actual_zone_id = self.world.zone_id_from_observation(observations)
        if actual_zone_id != self.zone.zone_id:
            return self.world.action('move', self.zone.zone_id), True
        return None, True

class GetItem(Option):

    def __init__(self, world:"crafting.world.World",
            item:Item,
            all_options: Dict[Union[int, str], Option],
            items_needed:List[List[tuple]],
            last_action: tuple,
            zones_id_needed=None, zones_properties_needed=None):

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
            self.zones_properties_needed = []

        self.last_action = last_action
        self.all_options = all_options

    def gather_items(self, observation, items_id_in_search):
        if len(self.items_needed) == 0:
            return None
        action_for_craft_option = [None for _ in self.items_needed]

        for i, craft_option in enumerate(self.items_needed):
            for item_id, quantity_needed in craft_option:
                if action_for_craft_option[i] is None:
                    item_slot = self.world.item_id_to_slot[item_id]
                    inventory_content = observation[:self.world.n_items]
                    has_enought = inventory_content[item_slot] >= quantity_needed
                    if not has_enought:
                        if item_id not in items_id_in_search:
                            get_item_option = self.all_options[item_id]
                            action_for_craft_option[i], _ = get_item_option(
                                observation, items_id_in_search=items_id_in_search
                            )
                        else:
                            action_for_craft_option[i] = "Not Feasable"
            if action_for_craft_option[i] is None:
                break

        need_an_action = all(action is not None for action in action_for_craft_option)
        any_craft_option = len(self.items_needed[0]) > 0
        if any_craft_option and need_an_action:
            feasable_actions = [
                action for action in action_for_craft_option
                if action is not None and action != "Not Feasable"
            ]
            if len(feasable_actions) > 0:
                return feasable_actions[0]
            return "Not Feasable"
        return None

    def move_to_any_zone_needed(self, observation):
        action_for_zone = [None for _ in self.zones_id_needed]
        for i, zone_id in enumerate(self.zones_id_needed):
            zone = self.world.zone_from_id[zone_id]
            action_for_zone[i], _ = self.all_options[str(zone)](observation)

        need_an_action = all(action is not None for action in action_for_zone)
        if len(self.zones_id_needed) > 0 and all(action is not None for action in action_for_zone):
            feasable_actions = [
                action for action in action_for_zone
                if action is not None and action != "Not Feasable"
            ]
            if len(feasable_actions) > 0:
                return feasable_actions[0]
            return "Not Feasable"
        return None


    def get_zone_property(self, zone_property, observation):
        props = self.world.properties_from_observation(observation)
        if zone_property not in props:
            get_property_option = self.all_options[zone_property]
            return get_property_option(observation)[0]

    def __call__(self, observations, greedy: bool=False, items_id_in_search=None):
        if items_id_in_search is None:
            items_id_in_search = []
        else:
            items_id_in_search = deepcopy(items_id_in_search)
        if self.item is not None:
            items_id_in_search.append(self.item.item_id)

        action = self.gather_items(observations, items_id_in_search)
        if action is not None:
            return action, False

        action = self.move_to_any_zone_needed(observations)
        if action is not None:
            return action, False

        for zone_property in self.zones_properties_needed:
            action = self.get_zone_property(zone_property, observations)
            if action is not None:
                return action, False

        return self.world.action(*self.last_action), True

    def get_graph(self) -> nx.DiGraph:

        def _add_predecessors(graph, prev_checks, node):
            if len(prev_checks) > 1:
                for pred in prev_checks:
                    graph.add_edge(pred, node, type='any', color='purple')
            elif len(prev_checks) == 1:
                graph.add_edge(prev_checks[0], node, type='conditional', color='green')

        graph = nx.DiGraph()
        prev_checks = []

        empty_node = None
        if len(self.items_needed) > 1:
            empty_node = ""
            graph.add_node(empty_node, type='empty', color='purple')

        for craft_option in self.items_needed: # Any of Craft options
            prev_check_in_option = None
            for item_id, quantity in craft_option:
                item = self.world.item_from_id[item_id]
                check_item = f"Has {quantity}\n{item} ?"
                get_item = f"Get\n{item}"
                graph.add_node(check_item, type='feature_check', color='blue')
                graph.add_node(get_item, type='option', color='orange')
                if prev_check_in_option is not None:
                    graph.add_edge(prev_check_in_option, check_item,
                        type='conditional', color='green')
                elif empty_node is not None:
                    graph.add_edge(empty_node, check_item, type='any', color='purple')
                graph.add_edge(check_item, get_item, type='conditional', color='red')
                prev_check_in_option = check_item
            if prev_check_in_option is not None:
                prev_checks.append(prev_check_in_option)

        prev_checks_zone = []
        for zone_id in self.zones_id_needed: # Any of the zones possibles
            zone = self.world.zone_from_id[zone_id]
            check_zone = f"Is in\n{zone} ?"
            go_to_zone = f"Go to\n{zone}"
            graph.add_node(check_zone, type='feature_check', color='blue')
            graph.add_node(go_to_zone, type='option', color='orange')
            if len(prev_checks) > 0:
                _add_predecessors(graph, prev_checks, check_zone)
            else:
                empty_node = ""
                graph.add_node(empty_node, type='empty', color='purple')
                graph.add_edge(empty_node, check_zone, type='any', color='purple')
            graph.add_edge(check_zone, go_to_zone, type='conditional', color='red')
            prev_checks_zone.append(check_zone)

        if len(prev_checks_zone) > 0:
            prev_checks = prev_checks_zone

        for prop in self.zones_properties_needed: # All properties needed
            check_prop = f"Zone {prop} ?"
            get_prop = f"Get {prop}"
            graph.add_node(check_prop, type='prop_check', color='blue')
            graph.add_node(get_prop, type='option', color='orange')
            if len(prev_checks) > 0:
                _add_predecessors(graph, prev_checks, check_prop)
            graph.add_edge(check_prop, get_prop, type='conditional', color='red')
            prev_checks = [check_prop]

        # Add last action
        action_type, obj_id = self.last_action
        if action_type == 'get':
            item = self.world.item_from_id[obj_id]
            last_node = f"Search\n{item}"
        elif action_type == 'craft':
            recipe = self.world.recipes_from_id[obj_id]
            last_node = f"Craft\n{recipe.outputs}"
        elif action_type == 'move':
            zone = self.world.zone_from_id[obj_id]
            last_node = f"Move to\n{zone}"
        else:
            raise ValueError(f'Unknowed action_type: {action_type}')

        graph.add_node(last_node, type='action', color='red')
        _add_predecessors(graph, prev_checks, last_node)
        return graph

    def draw_graph(self, ax):
        graph = self.get_graph()
        if len(list(graph.nodes())) > 0:
            compute_levels(graph)
            pos = option_layout(graph)

            nx.draw_networkx_nodes(
                graph, pos,
                ax=ax,
                node_color=[color for _, color in graph.nodes(data='color')],
            )

            nx.draw_networkx_labels(
                graph, pos,
                ax=ax,
                font_color='white',
            )

            nx.draw_networkx_edges(
                graph, pos,
                ax=ax,
                arrowsize=40,
                arrowstyle="->",
                edge_color=[color for _, _, color in graph.edges(data='color')]
            )

        return ax

    def __str__(self):

        def _add_text(string, text, do_return=False):
            if do_return:
                string += "\n"
            string += text
            return string

        string = ""
        if self.item is not None:
            string = _add_text(string, f"{self.item}\n")

        do_return = False
        if len(self.items_needed) > 0 and len(self.items_needed[0]) > 0:
            string = _add_text(string, f"  Required {self.items_needed}", do_return)
            do_return = True
        if len(self.zones_id_needed) > 0:
            string = _add_text(string, f"  Zones {self.zones_id_needed}", do_return)
            do_return = True
        if len(self.zones_properties_needed) > 0:
            string = _add_text(string, f"  Properties {self.zones_properties_needed}", do_return)
            do_return = True
        string = _add_text(string, f"  Last action {self.last_action}", do_return)
        return string
