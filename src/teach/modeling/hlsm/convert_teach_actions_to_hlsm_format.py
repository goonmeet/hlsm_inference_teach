import pdb
import json


filename = "/home/ubuntu/efs/teach/src/teach/meta_data_files/default_definitions.json"

action_definitions = json.load(open(filename, "r"))

IDX_TO_ACTION_TYPE = {}
NAV_ACTION_TYPES = []
INTERACT_ACTION_TYPES = []

for action in action_definitions["definitions"]["actions"]:
    IDX_TO_ACTION_TYPE[action["action_id"]] = action["action_name"]
    if action["action_type"] == "Motion":
        NAV_ACTION_TYPES.append(action["action_name"])

    if action["action_type"] == "ObjectInteraction":
        INTERACT_ACTION_TYPES.append(action["action_name"])

print(IDX_TO_ACTION_TYPE)

print(NAV_ACTION_TYPES)

print(INTERACT_ACTION_TYPES)
