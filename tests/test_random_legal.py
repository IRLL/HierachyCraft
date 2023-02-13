from crafting.env import CraftingEnv
from crafting.world import world_from_transformations

import pytest_check as check

from tests.envs import classic_env
import numpy as np


def random_legal_agent(observation, action_is_legal):
    action = np.random.choice(np.nonzero(action_is_legal)[0])
    return action


def test_random_legal_agent():
    named_transformations, start_zone, _, _, _ = classic_env()
    transformations = list(named_transformations.values())
    world = world_from_transformations(transformations, start_zone=start_zone)
    env = CraftingEnv(world, max_step=10)
    done = False
    observation = env.reset()
    total_reward = 0
    while not done:
        action_is_legal = env.actions_mask
        action = random_legal_agent(observation, action_is_legal)
        _observation, reward, done, _info = env.step(action)
        total_reward += reward

    check.greater_equal(total_reward, 0)
