from textarena.agents.basic_agents import STANDARD_GAME_PROMPT
from textarena.core import ObservationWrapper, Env, ObservationType
from clemcore.clemgame import GameMaster

from typing import Dict
import logging
logger = logging.getLogger(__name__)

class ClemObservationWrapper(ObservationWrapper):
    """
    ClemObservationWrapper is a custom observation wrapper for the TextArena environment.
    """
    
    def __init__(self, env: Env, game_master: GameMaster, num_players: int):
        super().__init__(env)
        self.game_master = game_master
        self.logged_observation_count = { i: 0 for i in range(num_players) }
    
    def _convert_obs_to_context(self, player_id):
        """
        Takes the observations the player hasn't seen yet, and prepares them as content in a context dict.
        """
        content = ""
        if self.logged_observation_count[player_id] == 0: # Initial call, add the standard game prompt
            content += STANDARD_GAME_PROMPT + "\n\n"

        observations = self.env.state.observations[player_id][self.logged_observation_count[player_id]:]
        self.logged_observation_count[player_id] += len(observations)
        for sender_id, message, observation_type in observations:
            assert isinstance(sender_id, int), f"first element of observation tuple should be int (sender_id), but is {type(sender_id)}"
            assert isinstance(message, str), f"second element of observation tuple should be str (message), but is {type(message)}"
            assert isinstance(observation_type, ObservationType), f"third element of observation tuple should be ObservationType, but is {type(observation_type)}"
            if not observation_type == ObservationType.PLAYER_ACTION:
                sender_name = self.env.state.role_mapping.get(sender_id, f"Player {sender_id}")
                content += f"[{sender_name}] {message}\n"
            else:
                if sender_id == player_id:
                    # skip own actions
                    continue
                # add messages for other players' actions
                sender_name = self.env.state.role_mapping.get(sender_id, f"Player {sender_id}")
                content += f"[{sender_name}] {message}\n"
        logger.info(f"Context for player {player_id}:\n{content.strip()}")
        return {'role': 'user', 'content': content}

    def observation(self, player_id: int) -> Dict:
        """
        env.get_observation() eventually calls this method, which calls _convert_obs_to_context(player_id).
        Thus, it returns a context dict for the player.
        """
        return self._convert_obs_to_context(player_id=player_id)