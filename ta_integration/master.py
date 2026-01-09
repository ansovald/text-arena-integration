from clemcore.clemgame import GameBenchmark
from clemcore.clemgame.registry import GameSpec
from clemcore.clemgame.metrics import BENCH_SCORE
from instancegenerator import generate_instances

import os

from submasters import *
from metrics import *

from typing import List, Dict, Tuple
import logging

# Local type aliases to avoid import issues with the framework. Copied from textarena/core.py
Message = Tuple[int, str]
Observations = Dict[int, List[Message]]

logger = logging.getLogger(__name__)

class TextArenaBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        logger.info(f"Initializing TextArenaBenchmark with game_spec: {game_spec}")
        super().__init__(game_spec)
        if 'master' in game_spec:
            master = game_spec['master']
        else:
            if game_spec['players'] == 1:
                master = 'SinglePlayerMaster'
            elif game_spec['players'] == 2:
                master = 'TwoPlayerMaster'
            else:
                raise ValueError("Multi-player games require a specified 'master' in the game spec.")
        if 'scorer' in game_spec:
            scorer = game_spec['scorer']
        else:
            if game_spec['players'] == 1:
                scorer = 'SinglePlayerScorer'
            elif game_spec['players'] == 2:
                scorer = 'TwoPlayerScorer'
            else:
                raise ValueError("Multi-player games require a specified 'scorer' in the game spec.")
        self.master_class = globals()[master]
        self.scorer_class = globals()[scorer]

        base_path = os.path.join(os.path.dirname(__file__), 'in')
        instance_file = os.path.join(base_path, game_spec['instances'] + '.json')
        print(f"Looking for instance file in {instance_file}")
        if not os.path.exists(instance_file):
            print("Instance file doesn't exist, generating...")
            game_spec = game_spec.__dict__
            generate_instances(**game_spec)

    def create_game_master(self, experiment, player_models):
        return self.master_class(self.game_spec, experiment, player_models)
    
    def create_game_scorer(self, experiment, game_instance):
        return self.scorer_class(self.game_spec['game_name'], experiment, game_instance)
    