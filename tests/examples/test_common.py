import pytest
import pytest_check as check

import os

from unified_planning.io import PDDLWriter

from crafting.render.human import render_env_with_human
from crafting.examples import EXAMPLE_ENVS
from crafting.examples.minicraft import (
    MiniCraftBlockedUnlockPickup,
    MiniCraftKeyCorridor,
)
from crafting.env import CraftingEnv


@pytest.mark.parametrize("env_class", EXAMPLE_ENVS)
def test_build_env(env_class):
    human_run = False
    env = env_class()
    if human_run:
        render_env_with_human(env)


@pytest.mark.parametrize("env_class", EXAMPLE_ENVS)
def test_pddl_solve(env_class):
    write = False
    env: CraftingEnv = env_class(max_step=200)
    problem = env.planning_problem()

    if write:
        writer = PDDLWriter(problem.upf_problem)
        pddl_dir = os.path.join("planning", "pddl", env.name)
        os.makedirs(pddl_dir, exist_ok=True)
        writer.write_domain(os.path.join(pddl_dir, "domain.pddl"))
        writer.write_problem(os.path.join(pddl_dir, "problem.pddl"))

    if isinstance(env, MiniCraftBlockedUnlockPickup):
        return  # Infinite loop for no reason ???

    done = False
    _observation = env.reset()
    while not done:
        action = problem.action_from_plan(env.state)
        _observation, _reward, done, _ = env.step(action)
    check.is_true(env.purpose.terminated)


KNOWN_TO_FAIL = [MiniCraftKeyCorridor, MiniCraftBlockedUnlockPickup]


@pytest.mark.parametrize("env_class", EXAMPLE_ENVS)
def test_can_solve(env_class):
    env: CraftingEnv = env_class(max_step=50)
    done = False
    observation = env.reset()
    for task in env.purpose.best_terminal_group.tasks:
        solving_behavior = env.solving_behavior(task)
        task_done = task.terminated
        while not task_done and not done:
            action = solving_behavior(observation)
            observation, _reward, done, _ = env.step(action)
            task_done = task.terminated
    if env_class in KNOWN_TO_FAIL:
        pytest.xfail("Known to fail on this environment")
    check.is_true(env.purpose.terminated)


@pytest.mark.parametrize("env_class", EXAMPLE_ENVS)
def test_requirements_graph(env_class):
    draw = False
    env: CraftingEnv = env_class()
    requirements = env.world.requirements
    requirements.graph
    if draw:
        import matplotlib.pyplot as plt

        width = max(requirements.depth, 10)

        fig, ax = plt.subplots()
        plt.tight_layout()
        fig.set_size_inches(width, 9 / 16 * width)
        requirements.draw(ax)

        requirements_dir = os.path.join("docs", "images", "requirements_graphs")
        os.makedirs(requirements_dir, exist_ok=True)

        filename = os.path.join(requirements_dir, f"{env.name}.png")
        if not os.path.exists(filename):
            fig.savefig(filename, dpi=80 * 16 / width, transparent=True)
        plt.close()
