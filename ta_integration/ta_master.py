"""
This module contains the default TextArena game master, scorer and player classes.
"""

from typing import Dict, Optional, Tuple, List
import abc
import random
import numpy as np
import textarena as ta
from clemcore.backends import Model
from clemcore.clemgame import Player, GameMaster
from clemcore.clemgame.registry import GameSpec
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE
import textarena.envs.registration as ta_env_reg
import logging
from typing import List, Dict, Optional, Tuple, Union

from clem_observation_wrapper import ClemObservationWrapper

module_logger = logging.getLogger(__name__)

class TextArenaPlayer(Player):
    """
    TextArenaPlayer is a custom player for the TextArena environment.
    It is subclassed for each player role defined in the game specification.
    """
    def __init__(self, model: Model, ta_player_id: int, master: GameMaster, custom_response: list = None):
        super().__init__(model=model)
        self.custom_response = custom_response if custom_response else ["No custom response specified."]
        self.ta_id = ta_player_id  # This is the player ID used by TextArena, which may differ from the GameMaster's player ID
        self.game_master = master  # Reference to the GameMaster for accessing game state and methods
    
    def _custom_response(self, context):
        return random.choice(self.custom_response)
    
    def perceive_context(self, context, *, log_event=True, memorize=True):
        self.game_master.played_in_round.add(self.ta_id)  # Mark this player as having played in the current turn
        self.game_master.last_player = self.ta_id  # Update the last player who played
        return super().perceive_context(context, log_event=log_event, memorize=memorize)

class TextArenaGameMaster(GameMaster):
    """
    TextArenaGameMaster is a custom game master for the TextArena environment.
    It inherits from the GameMaster class and implements the required methods.
    """
    
    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        module_logger.info(f"Initializing {self.__class__.__name__} for game {game_spec['game_name']}, experiment {experiment['name']}")
        super().__init__(game_spec=game_spec, experiment=experiment, player_models=player_models)
        self.game_name = game_spec['game_name']
        self.ta_env_id = None
        self.env = None
        self.players_by_id = { -1: "GM" }     # corresponds to the player IDs used by TextArena
        self.logged_observation_count = {}
        self.request_violation = False  # Flag to indicate if a request violation occurred in last turn
        self.played_in_round = set()  # Keep track of players who have played in the current turn
        self.last_player = None # Needed to detect possible request violation in last turn in _on_after_game()
        self.started = False  # Flag to indicate if the game has started
        self.done = False  # Flag to indicate if the game is done
        self.checked_observations = {}

    def setup(self, **kwargs):
        self.ta_env_id = kwargs.get("env_id")
        if kwargs.get("register_env", False) and not self.ta_env_id in ta_env_reg.ENV_REGISTRY:
            ta_env_reg.register(id=self.ta_env_id, entry_point=kwargs.get("entry_point"), **kwargs.get("env_specs", {}))
        random.seed(kwargs['seed'])
        np.random.seed(kwargs['seed'])
        self.started = True
        self.env = ta.make(env_id=self.ta_env_id)     
        self.env = ClemObservationWrapper(env=self.env, game_master=self, num_players=len(self.player_models))
        self._on_before_reset()
        self.env.reset(num_players=len(self.player_models), seed=kwargs['seed'])     # reset sets the initial prompts for each player
        player_specs = kwargs.get("player_specs", None)
        if not player_specs:
            player_specs = self._dummy_player_specs(n_players=len(self.player_models))
        for player_id, (player_spec, player_model) in enumerate(zip(player_specs, self.player_models)):
            # make a custom Player subclass for each player_spec, named after the role
            role = player_spec['role']
            custom_response = player_spec['custom_response']
            player_class = type(role, (TextArenaPlayer,), {})
            # Create an instance of the player class with the model
            player_instance = player_class(model=player_model, master=self, custom_response=custom_response, ta_player_id=player_id)
            self.add_player(player_instance, player_id=player_id)
        self._on_before_game()

    def _dummy_player_specs(self, n_players: int) -> List[Dict]:
        """
        Create dummy player specs with generic custom responses.
        """
        player_specs = []
        for i in range(n_players):
            player_specs.append({
                "role": f"Player {i}",
                "custom_response": None
            })
        return player_specs

    def _on_before_reset(self):
        """
        This method is called after the environment is created, but before it is reset.
        It can, e.g., be used to change variables in the env to make instances deterministic.
        """
        pass
    
    def _on_before_game(self):
        """
        This method is called after the environment is reset, but before the game starts.
        It can be used to set up the game state or perform any necessary initialization.
        """
        pass

    def has_started(self):
        return self.started
    
    def is_done(self):
        return self.done

    def add_player(self, player: Player, player_id: int):
        """
        Add a player to the game master.
        """
        player.register_many(self._loggers)
        self.players_by_id[player_id] = player
        player.name = f"Player {player_id}"        # TA player ID 0 is Player 1, etc.
        self.log_player(player.name, game_role=player.game_role, model_name=player.model.name)
        self.checked_observations[player_id] = 0

    def get_current_player(self) -> Optional[Player]:
        """
        Playpen needs this.
        Get the current player based on the environment's state.
        """
        return self.players_by_id[self.env.state.current_player_id]
    
    def get_context_for(self, player: Player) -> Dict:
        """
        Playpen needs this.
        Directly call the ClemObservationWrapper's observation method and bypass all other logic.
        """
        context = self.env.observation(player_id=player.ta_id)
        return context
    
    def observe(self) -> Tuple[Player, Dict]:
        player = self.get_current_player()
        context = self.get_context_for(player)
        return player, context
    
    def play(self):
        """
        Main play loop
        """
        done = False
        while not done:
            player, context = self.observe()
            response = player(context)
            done, info = self.step(response=response)

    def step(self, response: str) -> Tuple[Union[bool, List], Union[Dict, List]]:
        module_logger.info(f"Player {self.env.state.current_player_id} response: {response}")
        self.request_violation = False  # Reset request violation for this turn
        done, info = self.env.step(action=response)
        self.request_violation = self._check_move_validity()  # Check if the last player made an invalid move
        self.done = done
        if info:
            module_logger.info(f"Game info passed by TextArena's env.step(): {info}")
            self.log_to_self(type_='info', value=str(info))
        if not done:
            if self._start_next_round():
                self._prepare_next_round()
        else:
            self._after_game()
        return done, info
    
    def _start_next_round(self):
        """
        check if all players have played in the current turn and no request violation occurred.
        """
        return len(self.played_in_round) == len(self.players_by_id) - 1 and not self.request_violation

    def _prepare_next_round(self):
        self.played_in_round.clear()  # Clear the set for the next round
        self.log_next_round()

    def _check_move_validity(self):
        """
        Check if the last player made an invalid move.
        """
        player_id = self.env.state.current_player_id
        checked_observations = self.checked_observations[player_id]
        last_observations = self.env.state.observations[player_id][checked_observations:]
        self.checked_observations[player_id] = len(self.env.state.observations[player_id])  # Update the checked observations count
        for observation in last_observations:
            if observation and "attempted an invalid move" in str(observation[1]):
                self.count_request_violation()
                return True
        return False
    
    def _after_game(self):
        """
        This method is called after the game ends.
        It retrieves the rewards generated by env.close()
        """
        for logger in self._loggers:
            if hasattr(logger, 'interactions'):
                if 'meta' in logger.interactions:
                    logger.interactions['meta']['ta_version'] = ta.__version__ if hasattr(ta, '__version__') else "unknown"
                else:
                    module_logger.warning("Recorder meta information not found; cannot log TextArena version.")
        rewards = self.env.close()
        last_message = {'type': 'send message', 'content': '[GAME] ' + rewards[1][self.last_player]['reason']}
        last_move_invalid = rewards[1][self.last_player]['invalid_move']
        if last_move_invalid:
            self.count_request_violation()

        self.log_event(from_="GM", to=self.get_current_player().name, action=last_message)
        for player_id in self.players_by_id:
            if player_id == -1:
                continue
            player_rewards = rewards[1][player_id]
            player_rewards['reward'] = rewards[0][player_id]
            log_string = "\n".join([f"{key}: {value}" for key, value in player_rewards.items()])
            self.log_to_self(type_='rewards', value=log_string)
        self.log_key("ta_reward", rewards[0])
        self.log_key("ta_reward_details", rewards[1])
        module_logger.info(f"ta_reward: {rewards[0]}")
        module_logger.info(f"ta_reward_details: {rewards[1]}")
        self._on_after_game(rewards=rewards)
        self.log_game_end()

    @abc.abstractmethod
    def _on_after_game(self, **kwargs):
        """
        METRIC_SUCCESS, METRIC_LOSE and METRIC_ABORTED must be logged here (can be initialized with default values in prepare_metrics()).
        Called from _on_after_game() after env.close() and rewards have been retrieved.
        """
        pass

    def prepare_metrics(self) -> Dict[str, float]:
        """
        Returns default values for the metrics.
        """
        return {
            METRIC_ABORTED: 0,
            METRIC_SUCCESS: 0,
            METRIC_LOSE: 0
        }
