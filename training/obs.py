from typing import Any, Dict, List

import numpy as np
from rlgym.api import AgentID, ObsBuilder
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import (BLUE_TEAM, BOOST_LOCATIONS,
                                               ORANGE_TEAM)


class NectoObs(ObsBuilder[AgentID, np.ndarray, GameState, int]):
    _boost_locations = np.array(BOOST_LOCATIONS)
    _invert = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 4)
    _norm = np.array([1.] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4)

    def __init__(self, n_players=6, tick_skip=8):
        super().__init__()
        self.n_players = n_players
        self.demo_timers = None
        self.boost_timers = None
        self.current_state = None
        self.current_qkv = None
        self.current_mask = None
        self.tick_skip = tick_skip

    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.demo_timers = np.zeros(self.n_players)
        self.boost_timers = np.zeros(len(initial_state.boost_pad_timers))
        shared_info["previous_action"] = {agent_id: np.zeros(8) for agent_id in initial_state.cars.keys()}

    def get_obs_space(self, agent: AgentID) -> int:
        return None

    def _maybe_update_obs(self, state: GameState, shared_info: Dict[str, Any]):
        if self.boost_timers is None:
            self.reset(state, shared_info)
        else:
            self.current_state = state

        qkv = np.zeros((1, 1 + self.n_players + len(state.boost_pad_timers), 24))  # Ball, players, boosts

        # Add ball
        n = 0
        ball = state.ball
        qkv[0, 0, 3] = 1  # is_ball
        qkv[0, 0, 5:8] = ball.position
        qkv[0, 0, 8:11] = ball.linear_velocity
        qkv[0, 0, 17:20] = ball.angular_velocity

        # Add players
        n += 1
        demos = np.zeros(self.n_players)  # Which players are currently demoed
        for player in state.cars.values():
            if player.team_num == BLUE_TEAM:
                qkv[0, n, 1] = 1  # is_teammate
            else:
                qkv[0, n, 2] = 1  # is_opponent
            car_data = player.physics
            qkv[0, n, 5:8] = car_data.position
            qkv[0, n, 8:11] = car_data.linear_velocity
            qkv[0, n, 11:14] = car_data.forward
            qkv[0, n, 14:17] = car_data.up
            qkv[0, n, 17:20] = car_data.angular_velocity
            qkv[0, n, 20] = player.boost_amount
            #             qkv[0, n, 21] = player.is_demoed
            demos[n - 1] = player.demo_respawn_timer == 0  # Keep track for demo timer
            qkv[0, n, 22] = player.on_ground
            qkv[0, n, 23] = not player.has_flipped
            n += 1

        # Add boost pads
        n = 1 + self.n_players
        boost_pads = state.boost_pad_timers
        qkv[0, n:, 4] = 1  # is_boost
        qkv[0, n:, 5:8] = self._boost_locations
        qkv[0, n:, 20] = 0.12 + 0.88 * (self._boost_locations[:, 2] > 72)  # Boost amount
        #         qkv[0, n:, 21] = boost_pads

        # Boost and demo timers
        new_boost_grabs = (boost_pads == 1) & (self.boost_timers == 0)  # New boost grabs since last frame
        self.boost_timers[new_boost_grabs] = 0.4 + 0.6 * (self._boost_locations[new_boost_grabs, 2] > 72)
        self.boost_timers *= boost_pads  # Make sure we have zeros right
        qkv[0, 1 + self.n_players:, 21] = self.boost_timers
        self.boost_timers -= self.tick_skip / 1200  # Pre-normalized, 120 fps for 10 seconds
        self.boost_timers[self.boost_timers < 0] = 0

        new_demos = (demos == 1) & (self.demo_timers == 0)
        self.demo_timers[new_demos] = 0.3
        self.demo_timers *= demos
        qkv[0, 1: 1 + self.n_players, 21] = self.demo_timers
        self.demo_timers -= self.tick_skip / 1200
        self.demo_timers[self.demo_timers < 0] = 0

        # Store results
        self.current_qkv = qkv / self._norm
        mask = np.zeros((1, qkv.shape[1]))
        mask[0, 1 + len(state.cars):1 + self.n_players] = 1
        self.current_mask = mask

    def build_obs(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        actions = {}
        if self.boost_timers is None:
            return np.zeros(0)  # Obs space autodetect, make Aech happy
        self._maybe_update_obs(state, shared_info)

        for i, agent_id in enumerate(agents):
            player = state.cars[agent_id]
            invert = player.team_num == ORANGE_TEAM

            qkv = self.current_qkv.copy()
            mask = self.current_mask.copy()

            main_n = i + 1
            qkv[0, main_n, 0] = 1  # is_main
            if invert:
                qkv[0, :, (1, 2)] = qkv[0, :, (2, 1)]  # Swap blue/orange
                qkv *= self._invert  # Negate x and y values

            # TODO left-right normalization (always pick one side)

            q = qkv[0, main_n, :]
            q = np.expand_dims(np.concatenate((q, shared_info["previous_action"][agent_id]), axis=0), axis=(0, 1))
            # kv = np.delete(qkv, main_n, axis=0)  # Delete main? Watch masking
            kv = qkv

            # With EARLPerceiver we can use relative coords+vel(+more?) for key/value tensor, might be smart
            kv[0, :, 5:11] -= q[0, 0, 5:11]
            # return q, kv, mask
            actions[agent_id] = (q, kv, mask)
        return actions
