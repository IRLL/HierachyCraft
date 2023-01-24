# Crafting a meta-environment to simultate inventory managment
# Copyright (C) 2021-2023 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
import pytest_check as check
import numpy as np

from crafting.env import CraftingEnv, Item, Zone
from tests.check_array import check_np_equal


class TestCratingEnv:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.n_items = 5
        self.n_zones = 4
        self.n_transformations = 6

        self.items = [Item(str(i)) for i in range(self.n_items)]
        self.zones = [Zone(str(i)) for i in range(self.n_zones)]

        self.env = CraftingEnv(
            items=self.items,
            zones=self.zones,
            transformations=[],
        )

    def test_state_initialisation(self):
        player_inventory, position, zones_inventories = self.env.state
        expected_player_inventory = np.zeros(self.n_items)
        check_np_equal(player_inventory, expected_player_inventory)

        expected_position = np.zeros(self.n_zones)
        expected_position[0] = 1
        check_np_equal(position, expected_position)

        expected_zones_inventories = np.zeros((self.n_items, self.n_zones))
        check_np_equal(zones_inventories, expected_zones_inventories)