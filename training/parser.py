from typing import Dict, Any

import numpy as np

from rlgym.api import ActionParser, AgentID
from rlgym.rocket_league.api import GameState


class NectoAction(ActionParser[AgentID, np.ndarray, np.ndarray, GameState, int]):
    def get_action_space(self, agent: AgentID) -> int:
        return len(1944)

    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def parse_actions(self, actions: Dict[AgentID, np.ndarray], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        return actions
