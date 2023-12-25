from typing import Dict

import rlviser_py as rlviser
import RocketSim as rsim
from rlgym.api import AgentID, ObsType, RLGym
from rlgym.rocket_league.api import GameState


def check_rlviser_state(env: RLGym) -> Dict[AgentID, ObsType]:
    desired_state = rlviser.get_state_set()
    if desired_state is None:
        return
    
    for idx, pad in enumerate(env.transition_engine._arena.get_boost_pads()):
        pad_state = rsim.BoostPadState()
        pad_state.cooldown = desired_state[0][idx]
        pad.set_state(pad_state)

    ball_state = rsim.BallState()
    ball_state.pos = rsim.Vec(*desired_state[1][0])
    ball_state.rot_mat = rsim.RotMat(
        rsim.Vec(*desired_state[1][1][0]),
        rsim.Vec(*desired_state[1][1][1]),
        rsim.Vec(*desired_state[1][1][2])
    )
    ball_state.vel = rsim.Vec(*desired_state[1][2])
    ball_state.ang_vel = rsim.Vec(*desired_state[1][3])
    env.transition_engine._arena.ball.set_state(ball_state)

    desired_state[2].sort(key=lambda x: x[0])
    for i, car in enumerate(env.transition_engine._cars.values()):
        new_state = car.get_state()
        new_state.pos = rsim.Vec(*desired_state[2][i][1])
        new_state.rot_mat = rsim.RotMat(
            rsim.Vec(*desired_state[2][i][2][0]),
            rsim.Vec(*desired_state[2][i][2][1]),
            rsim.Vec(*desired_state[2][i][2][2])
        )
        new_state.vel = rsim.Vec(*desired_state[2][i][3])
        new_state.ang_vel = rsim.Vec(*desired_state[2][i][4])
        new_state.boost = desired_state[2][i][5]
        new_state.has_jumped = desired_state[2][i][6]
        new_state.has_double_jumped = desired_state[2][i][7]
        new_state.has_flipped = desired_state[2][i][8]
        new_state.demo_respawn_timer = desired_state[2][i][9]
        car.set_state(new_state)

    state = env.transition_engine._get_state()
    return env.obs_builder.build_obs(env.agents, state, env.shared_info)
