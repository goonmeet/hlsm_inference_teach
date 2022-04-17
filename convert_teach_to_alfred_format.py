import sys, os
import json
import argparse
from tqdm import tqdm

# default value is None
def convert_to_alfred(input_data, game_dir):
	example_dict = dict()
	game_data = json.load(open(os.path.join(game_dir, '{}.game.json'.format(input_data['game_id']))))

	example_dict['task_id'] = input_data['instance_id']        # (unique trajectory ID)
	example_dict['task_type'] = None     # (one of 7 task types)
	example_dict['pddl_params'] = {'object_target': "",   # (object)
								   'parent_target': "",     # (receptacle)
								   'mrecep_target': "",             # (movable receptacle)
								   "toggle_target": "",             # (toggle object)
								   "object_sliced": False}          # (should the object be sliced?)

	# missing
	# TODO: fill in this data from other files
	example_dict['turk_annotations'] = {}
	example_dict['turk_annotations']['anns'] =  [
				{'task_desc': game_data['tasks'][0]['desc'],                 # (goal instruction)
				'high_descs': ["",   # (list of step-by-step instructions)
							  ],               # (indexes aligned with high_idx)
				}]

	example_dict['images'] = [{"low_idx": action['action_idx'],                    # (low-level action index)
			   "high_idx": 0,                   # (high-level action index)
			   "image_name": image} for image, action in zip(input_data['driver_image_history'], input_data['driver_action_history'])]   # (image filename)

	example_dict['plan'] = {'high_pddl':
				[{"high_idx": 0,                         # (high-level subgoal index hardcoded to zero)
				 "discrete_action":
					 {"action": "Global single action",             # (discrete high-level action)
					  "args": [],    # (discrete params)
				 "planner_action": None}}],      # (PDDL action)

			'low_actions':
				[{"high_idx": 0,                          # (high-level subgoal index)
				 "discrete_action":
					 {"action": action['action_name'],          # (discrete low-level action)
					  "args": None}}    # (compressed pixel mask for interact action)
					for action in input_data['driver_action_history']],              # (THOR API command for replay)
		   }

	return example_dict

'''
Usage: python convert_teach_to_alfred_format.py --input-dir edh_instances/train --game-input-dir all_game_files/ --output-dir edh_alfred/train
'''

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input-dir', type=str, required=True, help='Teach dataset JSON file')
	parser.add_argument('--game-input-dir', type=str, required=True, help='Teach dataset JSON file')
	parser.add_argument('--output-dir', type=str, required=True, help='ALFRED dataset compatible JSON file')
	args = parser.parse_args()

	for filename in tqdm(os.listdir(args.input_dir)):
		input_data = json.load(open(os.path.join(args.input_dir, filename), 'r'))

		output_data = convert_to_alfred(input_data, args.game_input_dir)
		# print(output_data)
		filename_w = filename.replace('.json', '')
		os.makedirs(os.path.join(args.output_dir, filename_w))
		json.dump(output_data, open(os.path.join(args.output_dir, filename_w, 'traj_data.json'), 'w'))

	# print(output_data)
