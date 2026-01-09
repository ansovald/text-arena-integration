import os
import json
import regex as re
import logging

from clemcore.clemgame import GameInstanceGenerator
from textarena.envs.registration import ENV_REGISTRY

# Seed for reproducibility, to be passed to the TextArena environments
SEED = 525119131

logger = logging.getLogger(__name__)

class TextArenaInstanceGenerator(GameInstanceGenerator):
    """
    TextArenaInstanceGenerator is a custom instance generator for the TextArena environment.
    It just holds the configuration for the games, and passes the random seed to the TextArena environment,
    which handle the actual instance generation.
    """

    def __init__(self):
        super().__init__(os.path.dirname(__file__))
    
    def on_generate(self, seed: int, **kwargs):
        self.entry_point = kwargs.get("entry_point")
        assert self.entry_point is not None, "entry_point must be specified in the game config!"
        game_name = kwargs.get("game_name")
        experiments = kwargs.get("experiments")
        n_instances = kwargs.get("n_instances", 1)
        self.player_specs = kwargs.get("player_specs", None)

        if not experiments or isinstance(experiments, list):
            # load all games from the ENV_REGISTRY that use the specified entry point
            for ta_game in ENV_REGISTRY:
                if ENV_REGISTRY[ta_game].entry_point == self.entry_point and ta_game.endswith("-raw"):
                    experiment_name = ta_game[:-4]
                    logger.info(f"Checking experiment {experiment_name} for game {game_name}")
                    if isinstance(experiments, list) and experiment_name not in experiments:
                        continue
                    config = self.generate_config(env_id=ta_game, env_specs=ENV_REGISTRY[ta_game].kwargs)
                    experiment = self.add_experiment(experiment_name)
                    self.generate_instances(experiment, n_instances, config, seed)
        elif isinstance(experiments, dict):
            for experiment_name in experiments:
                experiment = self.add_experiment(experiment_name)
                config = self.generate_config(env_id=f"{game_name}-{experiment_name}", register_env=True, env_specs=experiments[experiment_name])
                self.generate_instances(experiment, n_instances, config, seed)
        else:
            raise ValueError("experiments must be either None, a list of experiment_name levels, or a dict of experiment configs.")
    
    def generate_config(self, **kwargs):
        config = {"entry_point": self.entry_point}
        # add any additional kwargs to the config
        for key, value in kwargs.items():
            config[key] = value
        if self.player_specs:
            config["player_specs"] = self.player_specs
        return config
    
    def generate_instances(self, experiment, n_instances: int, config: dict, seed: int=SEED):
        for i in range(n_instances):
            game_instance = self.add_game_instance(experiment, game_id=i)
            for key, value in config.items():
                game_instance[key] = value
            game_instance["seed"] = seed + i

def generate_instances(**game):
    TextArenaInstanceGenerator().generate(**game, filename=game["instances"] + '.json', seed=SEED)

if __name__ == "__main__":
    # load clemgame.json to get the games
    clemgame_registry = json.load(open(os.path.join(os.path.dirname(__file__), "clemgame.json"), "r"))

    for game in clemgame_registry:
        TextArenaInstanceGenerator().generate(**game, filename=game["instances"] + '.json', seed=SEED)
