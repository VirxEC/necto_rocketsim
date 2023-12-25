import time
from itertools import chain
from typing import Dict

import numpy as np
from rlgym.api import AgentID, RLGym
from rlgym.rocket_league import done_conditions, state_mutators
from rlgym.rocket_league.action_parsers import RepeatAction
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RLViserRenderer, RocketSimEngine

from training.agent import Agent
from training.obs import NectoObs
from training.parser import NectoAction
from training.state_setter import check_rlviser_state

RENDER = True
NORMAL_TICK_SKIP = 8
TICK_SKIP = 1 if RENDER else NORMAL_TICK_SKIP
STEP_REPEAT = NORMAL_TICK_SKIP if RENDER else 1

# Beta controls randomness:
# 1=best action, 0.5=sampling from probability, 0=random, -1=worst action, or anywhere inbetween
BETA = 1
TEAM_SIZE = 3
SPAWN_OPPONENTS = True


if __name__ == "__main__":
    env = RLGym(
        state_mutator=state_mutators.MutatorSequence(
            state_mutators.FixedTeamSizeMutator(blue_size=TEAM_SIZE, orange_size=TEAM_SIZE if SPAWN_OPPONENTS else 0),
            state_mutators.KickoffMutator()
        ),
        obs_builder=NectoObs(),
        action_parser=RepeatAction(NectoAction(), repeats=TICK_SKIP),
        reward_fn=GoalReward(),
        termination_cond=done_conditions.GoalCondition(),
        truncation_cond=done_conditions.AnyCondition(
            done_conditions.TimeoutCondition(timeout=300.),
            done_conditions.NoTouchTimeoutCondition(timeout=30.)
        ),
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer(tick_rate=120/TICK_SKIP)
    )

    agents: Dict[AgentID, Agent] = {}

    try:
        while True:
            obs = env.reset()
            for agent_id in env.agents:
                if agent_id not in agents:
                    agents[agent_id] = Agent()

            steps = 0

            t0 = time.time()

            while True:
                if RENDER:
                    new_obs = check_rlviser_state(env)
                    if new_obs is not None:
                        obs = new_obs

                actions = {agent_id: agents[agent_id].act(obs[agent_id], BETA)[0] for agent_id in env.agents}

                for _ in range(STEP_REPEAT):
                    obs, _, terminated_dict, truncated_dict = env.step(actions)

                    if RENDER:
                        env.render()
                        time.sleep(max(t0 + steps / 120 - time.time(), 0))
                    steps += 1

                env.shared_info["previous_action"] = actions

                if any(chain(terminated_dict.values(), truncated_dict.values())):
                    break
    except KeyboardInterrupt:
        pass
