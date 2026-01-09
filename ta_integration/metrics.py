from clemcore.clemgame import GameScorer
from typing import Dict
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE
import numpy as np
import logging
import abc

module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.DEBUG)


def init_metrics(numeric_reward=None) -> Dict[str, float]:
        """
        Returns default values for the metrics.
        """
        metrics = {
            METRIC_ABORTED: 0,
            METRIC_SUCCESS: 0,
            METRIC_LOSE: 0,
            BENCH_SCORE: np.nan
        }
        if numeric_reward is not None:
            if numeric_reward == -1:
                metrics[METRIC_ABORTED] = 1
                metrics[METRIC_LOSE] = 1
            elif numeric_reward == 1:
                metrics[METRIC_SUCCESS] = 1
                metrics[BENCH_SCORE] = 100
            else:
                metrics[BENCH_SCORE] = numeric_reward * 100
        return metrics

class TextArenaScorer(GameScorer):
    """
    Default scorer for the TextArena environment.
    """
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        # Interaction keys to extract from episode interactions to compute scores
        self.interaction_keys = []

    def get_auxiliaries(self, episode_interactions: Dict) -> Dict[str, float]:
        """
        Extracts auxiliary information from the episode interactions.
        """
        auxiliaries = {key: episode_interactions[key] for key in self.interaction_keys if key in episode_interactions}
        return auxiliaries

    def compute_episode_scores(self, episode_interactions: Dict):
        auxiliaries = self.get_auxiliaries(episode_interactions)
        bench_score = self.compute_bench_score(auxiliaries=auxiliaries)
        self.log_episode_score(BENCH_SCORE, bench_score)

    @abc.abstractmethod
    def compute_bench_score(self, auxiliaries: Dict):
        pass

class SinglePlayerScorer(TextArenaScorer):
    """
    Scorer for single-player games in TextArena.
    """
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.interaction_keys = [METRIC_SUCCESS, METRIC_ABORTED, METRIC_LOSE, "numeric_reward", "ta_reward"]

    def compute_bench_score(self, auxiliaries: Dict):
        """
        Computes basic benchmark score based on numeric_reward. Can be extended in subclasses.
        """
        self.log_episode_score("Numeric Reward", auxiliaries['numeric_reward'])
        self.log_episode_score("ta_reward", auxiliaries['ta_reward'])
        if auxiliaries[METRIC_ABORTED] == 1 or auxiliaries[METRIC_LOSE] == 1:
            return np.nan
        else:
            return auxiliaries['numeric_reward'] * 100

class TwoPlayerScorer(TextArenaScorer):
    """
    Scorer for two-player games in TextArena.
    """
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.interaction_keys = ["ta_reward"]
    
    def compute_bench_score(self, auxiliaries: Dict):
        """
        Compute basic benchmark score. Returns nan for now.
        """
        for key in auxiliaries:
            self.log_episode_score(key, auxiliaries[key])
        return np.nan

class HangmanScorer(SinglePlayerScorer):
    """
    Scorer for the Hangman game.
    """
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.interaction_keys += ["target_word", "lives_left", "lives"]

    def compute_bench_score(self, auxiliaries: Dict):
        bench_score = super().compute_bench_score(auxiliaries)
        life_score = (0.5 * auxiliaries['lives_left'] / auxiliaries['lives']) + 0.5  # life_score is in the interval [0.5, 1]
        self.log_episode_score('Life Score', life_score)
        if bench_score is not None:
            bench_score *= life_score
        return bench_score

class WordChainsScorer(TwoPlayerScorer):
    """
    Scorer for the Word Chains game.
    """
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.interaction_keys = [METRIC_ABORTED, METRIC_LOSE, "word_length_diff", "end_word_length"]
    
    def compute_bench_score(self, auxiliaries: Dict):
        if auxiliaries[METRIC_ABORTED] == 1 or auxiliaries[METRIC_LOSE] == 1:
            return np.nan
        else:
            word_length_diff = auxiliaries["word_length_diff"]
            max_word_length = 21  # only 66 of 263,689 (.025%) words in the dictionary are longer than 21 letters, so this is a reasonable upper limit for a perfect game
            main_score = min(1, (word_length_diff / max_word_length)) * 100
            return main_score
        
class BattleshipScorer(TextArenaScorer):
    """
    Scorer for the Battleship game.
    """
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.interaction_keys = [METRIC_ABORTED, METRIC_LOSE, "cell_counts", "ta_reward"]

    def compute_episode_scores(self, auxiliaries: Dict):
        module_logger.debug(f"{self.__class__.__name__}: scoring {auxiliaries['meta']['experiment_name']}_{str(auxiliaries['meta']['game_id']).zfill(5)}")
        self.log_episode_score("ta_reward", auxiliaries['ta_reward'])
        player_scores = {0: {}, 1: {}}
        total_cells = auxiliaries['cell_counts']['total_cells']
        total_ship_cells = auxiliaries['cell_counts']['total_ship_cells']
        module_logger.info(f"Total Cells: {total_cells}, Total Ship Cells: {total_ship_cells}")
        for player_id in range(2):
            cell_counts = auxiliaries['cell_counts'][str(player_id)]
            module_logger.info(f"Player {player_id} Cell Counts:\n{cell_counts}")
            ship_score = 1 - (cell_counts['remaining_ship_cells'] / total_ship_cells)
            hit_ratio = cell_counts['hits'] / (cell_counts['hits'] + cell_counts['misses']) if (cell_counts['hits'] + cell_counts['misses']) > 0 else 0
            other_player_id = (int(player_id) + 1) % 2
            player_scores[other_player_id]['ship_score'] = ship_score
            player_scores[other_player_id]['hit_score'] = hit_ratio * total_cells / total_ship_cells
        module_logger.info(f"Final Player Scores: {player_scores}")
        self.log_episode_score('Player 0 Ship Score', player_scores[0]['ship_score'])
        self.log_episode_score('Player 0 Hit Score', player_scores[0]['hit_score'])
        self.log_episode_score('Player 1 Ship Score', player_scores[1]['ship_score'])
        self.log_episode_score('Player 1 Hit Score', player_scores[1]['hit_score'])
        mean_ship_score = (player_scores[0]['ship_score'] + player_scores[1]['ship_score']) / 2
        mean_hit_score = (player_scores[0]['hit_score'] + player_scores[1]['hit_score']) / 2
        self.log_episode_score('Mean Ship Score', mean_ship_score)
        self.log_episode_score('Mean Hit Score', mean_hit_score)
        # if auxiliaries[METRIC_ABORTED] == 1 or auxiliaries[METRIC_LOSE] == 1:
        #     self.log_episode_score(BENCH_SCORE, np.nan)
        # else:
        self.log_episode_score(BENCH_SCORE, mean_ship_score * 100)