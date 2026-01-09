from clemcore.clemgame import GameRegistry, GameSpec
from clemcore.backends import ModelSpec
from clemcore.cli import run
from pathlib import Path
from ta_integration.instancegenerator import generate_instances
import os
import json
from typing import Union, Dict, List

import logging

import argparse

logger = logging.getLogger("determinism_test")
logger.setLevel(logging.INFO)
OUT_FILE = 'determinism_results.json'

def create_test_instance(game_spec):
    game_path = game_spec.game_path
    instances = game_spec.instances
    instance_file = os.path.join(game_path, 'in', f'{instances}.json')
    if not os.path.exists(instance_file):
        generate_instances(**game_spec.__dict__)
    # load instance file
    with open(instance_file, 'r') as f:
        instances = json.load(f)

    # get first instance from first experiment
    first_instance = instances['experiments'][0]['game_instances'][0]

    test_instances = {'experiments': [
        {
            'name': 'determinism_test',
            'game_instances': [first_instance]
        }
    ]}
    
    test_instance_file = os.path.join(game_path, 'in', f'tmp_{game_spec.game_name}.json')
    with open(test_instance_file, 'w') as f:
        json.dump(test_instances, f, indent=4)
    return f'tmp_{game_spec.game_name}', test_instance_file

def test_determinism(game_selector: Union[str, Dict, GameSpec], model: str):
    game_specs = GameRegistry.from_directories_and_cwd_files().get_game_specs_that_unify_with(game_selector)

    models = [model]
    model_specs = ModelSpec.from_strings(models)
    gen_args = {
        'temperature': 0.0,
        'max_tokens': 300
    }

    game_spec = game_specs[0]
    print(f"Testing determinism for game {game_spec.game_name} with model {model_specs[0].model_name}")
    
    # initialize path list for cleanup
    tmp_paths = []

    # create test instance file
    test_instance_file, test_instance_path = create_test_instance(game_spec)
    tmp_paths.append(test_instance_path)
    interaction_files = []

    for i in range(2):
        results_path = Path(f"determinism_tests_{i}")
        run(game_selector=game_spec, 
            model_selectors=model_specs, 
            gen_args=gen_args, 
            results_dir_path=results_path,
            instances_filename=test_instance_file)
        interaction_files.append(os.path.join(results_path, f"{models[0]}-t0.0", game_spec.game_name, 'determinism_test', 'instance_00000', 'interactions.json'))
    # check if both files exist
    if not all(os.path.exists(f) for f in interaction_files):
        print(f"One of the interaction files does not exist for game {game_spec.game_name}.")
        raise FileNotFoundError("Interaction files missing.")

    interactions = []
    for f in interaction_files:
        # load file as txt, delete all lines that contain 'timestamp'
        with open(f, 'r') as file:
            lines = file.readlines()
        cleaned_lines = [line for line in lines if 'timestamp' not in line]
        interactions.append('\n'.join(cleaned_lines))

    # compare both interaction files
    test_passed = interactions[0] == interactions[1]
    if test_passed:
        logger.info(f"Determinism test passed for game {game_spec.game_name}.")
        for path in tmp_paths:
            if os.path.isdir(path):
                # remove directory and all its contents
                for root, dirs, files in os.walk(path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                        logger.debug(f"Removed file: {os.path.join(root, name)}")
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                        logger.debug(f"Removed directory: {os.path.join(root, name)}")
            elif os.path.isfile(path):
                os.remove(path)
                logger.debug(f"Removed file: {path}")
    else:
        logger.error(f"Determinism test failed for game {game_spec.game_name}! Keeping temporary files for inspection: {tmp_paths}")

    return game_spec.game_name, model, test_passed


def check_games(model: str, games: Union[List[str], str] = None, out_file: str = OUT_FILE, overwrite: bool = False):
    game_registry = GameRegistry.from_directories_and_cwd_files()
    game_specs = []
    if games is not None:
        if isinstance(games, str):
            games = [games]
        for game in games:
            specs = game_registry.get_game_specs_that_unify_with(game)
            game_specs.extend(specs)
    else:
        game_specs = game_registry.get_game_specs()
    # load out_file, if it exists, to get already tested games
    existing_results = {}
    if out_file and os.path.exists(out_file):
        with open(out_file, 'r') as f:
            existing_results = json.load(f)
    for game_spec in game_specs:
        if not overwrite and game_spec.game_name in existing_results and model in existing_results[game_spec.game_name]:
            logger.info(f"Skipping already tested game {game_spec.game_name} for model {model}.")
            continue
        try:
            game_name, test_model, test_passed = test_determinism(game_spec, model=model)
        except Exception as e:
            logger.error(f"Error testing game {game_spec.game_name} with model {model}: {e}")
            continue
        if out_file:
            if game_name not in existing_results:
                existing_results[game_name] = {}
            existing_results[game_name][test_model] = test_passed
            with open(out_file, 'w') as f:
                json.dump(existing_results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Check determinism of TA games with a specified model.")
    parser.add_argument('-m', '--model', type=str, required=True, help='Model to use for testing determinism.')
    parser.add_argument('-of', '--out_file', type=str, default=OUT_FILE, help='Output file to store results.')
    parser.add_argument('-ow', '--overwrite', action='store_true', help='Overwrite existing results in the output file.')
    parser.add_argument('-g', '--game', type=str, default=None, help='Specific game to test. If not provided, all games will be tested.')
    args = parser.parse_args()
    print(args)
    check_games(model=args.model, games=args.game, out_file=args.out_file, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
