import textarena.envs.registration as ta_env_reg
import textarena as ta
from clemcore.clemgame import GameRegistry
import os
import json
import inspect
import textarena.state as ta_state
from textarena.core import ObservationType
import time
import logging

logger = logging.getLogger("clem-ta.create_notes")
logger.setLevel(logging.INFO)

CURRENT_DATE = time.strftime("%Y-%m-%d")
NOTE_DIR = os.path.join(os.path.dirname(__file__), 'notes_and_templates')

# get all state classes in textarena.state
STATE_CLASSES = [name for name, _ in inspect.getmembers(ta_state, inspect.isclass) if name.endswith('State')]

# Multiplayer games differ in number of players. If we integrate them later, we have to figure out how to handle that.
PLAYER_NUMBERS = {
    'SinglePlayerState': 1,
    'TwoPlayerState': 2
}

def get_latest_game_info_file():
    # look for latest game info file in NOTE_DIR
    game_info_files = [f for f in os.listdir(NOTE_DIR) if f.startswith('ta_game_info_') and f.endswith('.json')]
    if not game_info_files:
        return None
    # sort files by date in filename
    game_info_files.sort(reverse=True)
    latest_file = game_info_files[0]
    return latest_file

LATEST_GAME_INFO_FILE = get_latest_game_info_file()

def create_game_info(game_info_file: str = None):
    """
    Create a JSON file with information about all TextArena games listed in `ta_env_reg.ENV_REGISTRY`.
    File will be saved as `ta_game_info_YYYY-MM-DD-ID.json` in the `notes_and_templates` directory, 
    where `ID` is a unique identifier to avoid overwriting existing files on the same day.
    If a game is already integrated in `clemgame.json`, it is marked as 'integrated'.
    If a game is not integrated, check if
        1) it uses "SinglePlayerState" or "TwoPlayerState" (multiplayer games are skipped for now)
        2) it uses an LLM jury (or OpenRouterAgent) -> mark as 'excluded'
        3) it can be set up without errors -> mark as 'not tested' (else 'excluded')
        4) try to extract possible commands from the initial observation prompt

    Additionally, adds prior notes from older `ta_game_info_YYYY-MM-DD-ID.json` files, 
    and marks if the game was tested with a model before.

    The resulting JSON file contains the following structure:
    {
        "entry_point": {
            "game_ids": [list of game ids],
            "status": "integrated" | "not tested" | "excluded",
            "model_tested": false,
            "prior_model_tests": {              # Only if game was tested before
                "YYYY-MM-DD": true,
                ...
            },
            "notes": "",
            "prior_notes": {              # Only if notes exist from older files
                "YYYY-MM-DD": "note text",
                ...
            },
            "description": "",
            "num_players": 1 | 2 | "multi",
            "possible_commands": [list of commands]
        },
        ...
    }
    """

    older_game_info, older_tests, file_id = get_old_notes()
    if not game_info_file:
        # if no specific file name is given, create a new one with unique ID
        game_info_file = f'ta_game_info_{CURRENT_DATE}-{file_id}.json'

    # Load game registry from clemgame.json files in current directory and subdirectories
    game_registry = GameRegistry.from_directories_and_cwd_files()
    integrated_entry_points = {}
    for game_spec in game_registry.get_game_specs():
        # TextArena games have 'entry_point' field
        if 'entry_point' in game_spec:
            entry_point = game_spec['entry_point']
            if entry_point not in integrated_entry_points:
                integrated_entry_points[entry_point] = game_spec
    print(f"Found {len(integrated_entry_points)} integrated entry points in `clemgame.json`.")

    excluded_entry_points = set()
    integratable_entry_points = set()
    game_info = {}
    for key, value in ta_env_reg.ENV_REGISTRY.items():
        game_id = value.id
        entry_point = value.entry_point
        if entry_point not in game_info:
            game_info[entry_point] = {
                'game_ids': [],
                'status': 'not tested',
                'model_tested': False,
                'prior_model_tests': older_tests.get(entry_point, {}),
                'notes': '',
                'prior_notes': older_game_info.get(entry_point, {}),
                'description': ''
            }
        # # add prior notes from older game info files
        # if entry_point in older_game_info:
        #     game_info[entry_point]['prior_notes'] = older_game_info[entry_point]
        if game_id.endswith("-raw"):
            # We only care about raw versions without default wrappers
            game_info[entry_point]['game_ids'].append(game_id)
        else:
            continue
        if entry_point in integrated_entry_points:
            # entry point already integrated, skip and add game description from clemgame.json
            game_info[entry_point]['status'] = 'integrated'
            game_info[entry_point]['description'] = integrated_entry_points[entry_point]['description']
        if entry_point in integratable_entry_points | excluded_entry_points:
            # entry point already tested, skip
            continue
        if 'jury_class' in value.kwargs:
            # game uses LLM jury, skip
            logger.info(f"Game {key} uses jury class, skipping.")
            game_info[entry_point]['status'] = 'excluded'
            game_info[entry_point]['notes'] += 'uses jury class; '
            excluded_entry_points.add(entry_point)
            continue

        # find the file path of the module
        module_name = entry_point.split(":")[0]
        try:
            # Try importing the module
            module = __import__(module_name, fromlist=[''])
            module_file = module.__file__
            # check if the file contains 'OpenRouterAgent', i.e., uses non-player LLM agent
            module_content = open(module_file).read()
            if 'OpenRouterAgent' in module_content:
                logger.info(f"Game {key} uses OpenRouterAgent, skipping.")
                excluded_entry_points.add(value.entry_point)
                game_info[entry_point]['status'] = 'excluded'
                game_info[entry_point]['notes'] += 'uses OpenRouterAgent; '
                continue
            # check which state class is used in the module
            for state_class in STATE_CLASSES:
                if state_class in module_content:
                    game_info[entry_point]['state_class'] = state_class
                    break
        except Exception as e:
            logger.info(f"Excluding {key} due to import error: {e}")
            excluded_entry_points.add(entry_point)
            game_info[entry_point]['status'] = 'excluded'
            game_info[entry_point]['notes'] += f'import error: {e}; '
            continue

        # we default to "multi" unless we can determine the number of players
        num_players = "multi"
        if game_info[entry_point].get('state_class') in PLAYER_NUMBERS:
            num_players = PLAYER_NUMBERS[game_info[entry_point]['state_class']]
        game_info[entry_point]['num_players'] = num_players

        # try setting up the environment
        try:
            env = ta.make(env_id=game_id)
            env.reset(num_players=num_players)

            player_id, observation = env.get_observation()
            # Make sure observation conforms to expected structure
            for sender_id, message, observation_type in observation:
                assert isinstance(sender_id, int), f"first element of observation tuple should be int (sender_id), but is {type(sender_id)}"
                assert isinstance(message, str), f"second element of observation tuple should be str (message), but is {type(message)}"
                assert isinstance(observation_type, ObservationType), f"third element of observation tuple should be ObservationType, but is {type(observation_type)}"
            prompt = observation[0][1]

            # find all substrings in prompt that are enclosed in square brackets; this should cover most commands
            commands = []
            start = prompt.find('[')
            while start != -1:
                end = prompt.find(']', start)
                if end != -1:
                    command = prompt[start:end+1]
                    if command not in commands:
                        commands.append(command)
                    start = prompt.find('[', end)
                else:
                    break
            if commands:
                game_info[entry_point]['possible_commands'] = commands
            
        except Exception as e:
            logger.info(f"Couldn't set up env for {key}: {e}")
            game_info[entry_point]['status'] = 'excluded'
            game_info[entry_point]['notes'] += f'couldn\'t set up env: `{e}`; '
            excluded_entry_points.add(entry_point)
            continue
        
        # if we reach this point, the entry point should be integratable
        integratable_entry_points.add(entry_point)
    
    # for each entry_point in game_info, check if 'game_ids' is empty. 
    # If so, mark as excluded (no `-raw` version without default wrappers available)
    for entry_point, info in game_info.items():
        if not info['game_ids']:
            info['status'] = 'excluded'
            info['notes'] += 'no `-raw` version available; '
            excluded_entry_points.add(entry_point)
            if entry_point in integratable_entry_points:
                integratable_entry_points.remove(entry_point)

    n_total_entry_points = len(integratable_entry_points) + len(excluded_entry_points)
    logger.info(f"""
                Total entry points:        {n_total_entry_points}
                Integrated entry points:   {len(integrated_entry_points)}
                Integratable entry points: {len(integratable_entry_points)}
                Excluded entry points:     {len(excluded_entry_points)}""")
    
    # sort game_info alphabetically by entry_point
    game_info = dict(sorted(game_info.items()))

    # save game_info to json file
    output_file = os.path.join(NOTE_DIR, game_info_file)
    with open(output_file, 'w') as f:
        json.dump(game_info, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved game info to {output_file}.")
    print(f"Saved game info to {output_file}.")

def get_old_notes():
    # look for older game info files 
    # load notes and test status and the number of files with today's date (to create unique file id)
    older_game_info_files = [f for f in os.listdir(NOTE_DIR) if f.startswith('ta_game_info_') and f.endswith('.json')]
    older_game_info = {}
    older_tests = {}
    files_with_current_date = [f for f in older_game_info_files if f[len('ta_game_info_'):-len('.json')].startswith(CURRENT_DATE)]
    for older_file in older_game_info_files:
        older_file_path = os.path.join(NOTE_DIR, older_file)
        try:
            older_data = json.load(open(older_file_path, 'r'))
            # Get the date from the filename
            date_id = older_file[len('ta_game_info_'):-len('.json')]

            for entry_point, info in older_data.items():
                if entry_point not in older_game_info:
                    older_game_info[entry_point] = {}
                if 'notes' in info and info['notes']:
                    older_game_info[entry_point][date_id] = f"{info['status']}: {info['notes']}"
                    # sort older notes by date
                    older_game_info[entry_point] = dict(sorted(older_game_info[entry_point].items()))
                    # Check if values are unique, else only keep the oldest note
                    unique_notes = {}
                    for k, v in older_game_info[entry_point].items():
                        if v not in unique_notes.values():
                            unique_notes[k] = v
                    older_game_info[entry_point] = unique_notes
                model_tested = info.get('model_tested', False)
                if model_tested:
                    if entry_point not in older_tests:
                        older_tests[entry_point] = {}
                    older_tests[entry_point][date_id] = model_tested
                    # sort older tests by date
                    older_tests[entry_point] = dict(sorted(older_tests[entry_point].items()))
        except Exception as e:
            logger.error(f"Could not load older game info file {older_file_path}: {e}")
    return older_game_info, older_tests, len(files_with_current_date)

def create_clemgame_templates(game_info_file: str = LATEST_GAME_INFO_FILE):
    game_info_file = os.path.join(NOTE_DIR, game_info_file)
    try:
        game_info = json.load(open(game_info_file, 'r'))
    except Exception as e:
        logger.error(f"Could not load note file {game_info_file}: {e}")
        return
    clemgame_list = []
    for entry_point, info in game_info.items():
        if info['status'] == 'not tested':
            game_id = info['game_ids'][0].split("-")[0]
            # convert from camel case to snake case
            game_id = ''.join(['_'+c.lower() if c.isupper() else c for c in game_id]).lstrip('_')
            game_name = 'ta_' + game_id
            num_players = info['num_players']
            clemgame_entry = {
                "game_name": game_name,
                "description": info.get('description', ''),
                "entry_point": entry_point,
                "n_instances": 1,
                "instances": 'in_' + game_id,
                "players": num_players,
            }
            if 'notes' in info and info['notes']:
                clemgame_entry['notes'] = info['notes']
            clemgame_list.append(clemgame_entry)
    output_file = os.path.join(NOTE_DIR, f'clemgame_templates.json')
    with open(output_file, 'w') as f:
        json.dump(clemgame_list, f, indent=4, ensure_ascii=False)

def get_stats(game_info_file: str = LATEST_GAME_INFO_FILE):
    game_info_file = os.path.join(NOTE_DIR, game_info_file)
    try:
        game_info = json.load(open(game_info_file, 'r'))
    except Exception as e:
        logger.error(f"Could not load note file {game_info_file}: {e}")
        return
    status_counts = {}
    total_counts = {}
    model_tested_counts = 0
    for entry_point, info in game_info.items():
        num_players = info.get('num_players', 'unknown')
        status = info['status']
        if num_players not in status_counts:
            status_counts[num_players] = {}
        if status not in status_counts[num_players]:
            status_counts[num_players][status] = 0
        if status not in total_counts:
            total_counts[status] = 0
        status_counts[num_players][status] += 1
        total_counts[status] += 1
    print("Overall status counts:")
    for status, count in total_counts.items():
        print(f"{status:20}{count}")
    print(f"{'total games':20}{len(game_info)}")
    print("\nStatus counts by number of players:")
    for num_players in status_counts:
        print(f"\n{'Number of players:':20}{num_players}")
        for status, count in status_counts[num_players].items():
            print(f"{status:20}{count}")


def create_game_list(out_file: str = os.path.join(NOTE_DIR, 'ta_game_list.txt')):
    """
    Creates a text file with the game_name of all TextArena games listed in `clemgame.json`.
    Can be copied into a bash script for batch processing.
    """
    game_registry = GameRegistry.from_directories_and_cwd_files()
    game_names = []
    for game_spec in game_registry.get_game_specs():
        if 'game_name' in game_spec:
            game_names.append(game_spec.game_name)
    # save as `games = (...)`
    bash_string = "games=(\n"
    for game_name in game_names:
        bash_string += f'  "{game_name}"\n'
    bash_string += ")"
    print(bash_string)
    with open(out_file, 'w') as f:
        f.write(bash_string + '\n')
    logger.info(f"Saved game list to {out_file}.")
    print(f"Saved game list to {out_file}.")

if __name__ == "__main__":
    # add argparse to allow specifying which function to run
    import argparse
    parser = argparse.ArgumentParser(description="TextArena integration utilities.")
    parser.add_argument('--game_info', action='store_true', help="Create game info JSON file.")
    parser.add_argument('--clemgame_templates', action='store_true', help="Create clemgame.json templates for unintegrated games.")
    parser.add_argument('--get_stats', action='store_true', help="Get statistics from game info JSON file.")
    parser.add_argument('--integrated_games', action='store_true', help="Create text file with list of integrated game names.")
    args = parser.parse_args()

    if args.game_info:
        create_game_info()
    if args.clemgame_templates:
        create_clemgame_templates()
    if args.get_stats:
        get_stats()
    if args.integrated_games:
        create_game_list()
    if not (args.game_info or args.clemgame_templates or args.get_stats or args.integrated_games):
        parser.print_help()
