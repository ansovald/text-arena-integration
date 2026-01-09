"""
Subclasses of TextArena's GameMaster to handle scoring
and functionalities to ensure deterministic behavior for specific games
"""
from ta_master import TextArenaGameMaster
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE
from typing import Dict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def reward_for_player(rewards: list[Dict], player_id: int=0) -> float:
    """
    Args:
        rewards: list of dicts as passed by ta_env.close()
        player_id: ID of the player for which to extract the reward
    Returns: 
        numeric_reward for the player
    """
    assert player_id in rewards[0], f"Player ID {player_id} not found in rewards! {rewards[0].keys()}"
    assert player_id in rewards[1], f"Player ID {player_id} not found in other_rewards! {rewards[1].keys()}"
    numeric_reward = rewards[0][player_id]
    # invalid_move = rewards[1][player_id]['invalid_move']
    # if invalid_move:
    #     numeric_reward = -1  # Bypass the float reward and sets it to -1, as described on TA website
    return numeric_reward

class SinglePlayerMaster(TextArenaGameMaster):
    """
    Master class for single-player games in TextArena.
    It handles basic scoring and logging functionalities.
    """
    def _on_after_game(self, **kwargs):
        rewards = kwargs.get('rewards', {})
        numeric_reward = reward_for_player(rewards, player_id=0)
        self.log_key('numeric_reward', numeric_reward)
        metrics = self.prepare_metrics(numeric_reward)
        for key, value in metrics.items():
            self.log_key(key, value)

    def prepare_metrics(self, numeric_reward=None) -> Dict[str, float]:
        """
        Returns default values for the metrics.
        """
        metrics = super().prepare_metrics()
        if numeric_reward:
            if numeric_reward == -1:
                metrics[METRIC_ABORTED] = 1
                metrics[METRIC_LOSE] = 1
            elif numeric_reward == 1:
                metrics[METRIC_SUCCESS] = 1
        return metrics
    
class MinesweeperMaster(SinglePlayerMaster):
    """
    BENCH_SCORE is here the percentage of cells cleared by 
    the player, excluding the ones revealed in the first turn.
    """
    def _on_before_game(self):
        """
        Mines are only distributed with the first move, so we first reveal the 
        center cell to ensure the game is always initialized deterministically.
        """
        rows = self.env.rows
        cols = self.env.cols
        x = rows // 2
        y = cols // 2
        faux_response = f"[{x} {y}]"
        player, context = self.observe()
        player.perceive_context(context)
        player.perceive_response(faux_response)
        self.step(faux_response)

class WordleMaster(SinglePlayerMaster):
    """
    Make sure averaged perceptron tagger from NLTK is downloaded
    Not sure why this threw an error, since the TextArena environment 
    should download it by itself, but this seems to fix it.
    """
    def __init__(self, *args, **kwargs):
        import nltk
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        super().__init__(*args, **kwargs)

class TwoPlayerMaster(TextArenaGameMaster):
    """
    Master class for competitive two-player games in TextArena.
    It handles basic scoring and logging functionalities.
    """
    def _on_after_game(self, **kwargs):
        rewards = kwargs.get('rewards', {})
        last_move_invalid = rewards[1][self.last_player]['invalid_move']
        
        if last_move_invalid:
            self.log_key(METRIC_ABORTED, 1)
            self.log_key(METRIC_SUCCESS, 0)
            self.log_key(METRIC_LOSE, 1)
        else:
            self.log_key(METRIC_ABORTED, 0)
            self.log_key(METRIC_SUCCESS, 1)
            self.log_key(METRIC_LOSE, 0)
    
class BattleshipMaster(TwoPlayerMaster):
    """
    Master class for the Battleship game in TextArena.
    It handles basic scoring and logging functionalities.
    """
    def _on_after_game(self, **kwargs):
        super()._on_after_game(**kwargs)
        boards = {}
        for player_id in self.env.state.game_state['board']:
            board = self.env.state.game_state['board'][player_id]
            board = "\n".join("".join(f"{cell}" for cell in board[i]) for i in range(len(board)))
            boards[player_id] = board
        cell_counts = {
            'total_cells': self.env.grid_size ** 2,
            'total_ship_cells': sum(value for value in self.env.ships.values())
        }
        for player_id in boards:
            # count the number of 'X' and 'O' on the board
            hits = boards[player_id].count('X')
            misses = boards[player_id].count('O')
            water = boards[player_id].count('~')
            remaining_ships = self.env.grid_size ** 2 - (hits + misses + water)
            cell_counts[player_id] = {
                'hits': hits,
                'misses': misses,
                'water': water,
                'remaining_ship_cells': remaining_ships
            }
        self.log_key('cell_counts', cell_counts)


class WordChainsMaster(TextArenaGameMaster):
    def _on_before_reset(self):
        # Sort word_list of self.env to ensure deterministic sampling of start word
        self.env.word_list.sort()
    
    def _on_before_game(self):
        self.start_word = self.env.state.game_state['current_word']
        self.log_key('start_word', self.start_word)
        self.log_key('start_word_length', len(self.start_word))

    def _on_after_game(self, **kwargs):
        end_word = self.env.state.game_state['current_word']
        self.log_key('end_word', end_word)
        self.log_key('end_word_length', len(end_word))
        word_length_diff = len(end_word) - len(self.start_word)
        self.log_key('word_length_diff', word_length_diff)
        
        if word_length_diff == 0:
            self.log_key(METRIC_ABORTED, 1)
            self.log_key(METRIC_SUCCESS, 0)
            self.log_key(METRIC_LOSE, 1)
        else:
            self.log_key(METRIC_ABORTED, 0)
            self.log_key(METRIC_SUCCESS, 1)
            self.log_key(METRIC_LOSE, 0)