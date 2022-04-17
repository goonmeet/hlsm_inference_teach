from typing import List, Dict

import copy

from lgp.env.teach.teach_action import TeachAction
from lgp.env.teach.teach_subgoal import TeachSubgoal


class TeachHLPreproc:
    def __init__(self):
        ...

    def process_sample(self, sample):
        try:
            del sample["observation"].privileged_info._world_state
            del sample["observation"].privileged_info._task
        except (AttributeError, KeyError) as e:
            pass
        try:
            del sample["eventual_action_observation"].privileged_info._world_state
            del sample["eventual_action_observation"].privileged_info._task
        except (AttributeError, KeyError) as e:
            pass

    def process(self, rollout):
        # Delete information in the rollout that's not used for training purposes.
        for sample in rollout:
            self.process_sample(sample)
        return rollout


class TeachHLChunkingStrategy:

    def __init__(self):
        ...

    def is_sequence_terminal(self, action: TeachAction):
        return action.action_type in TeachAction.get_interact_action_list() or action.is_stop()

    def include_chunk(self, action: TeachAction):
        return True

    def ll_to_hl(self, samples: List[Dict], start_idx : int):
        rollout_out = []

        last_action = samples[-1]["action"]
        last_observation = samples[-1]["observation"]

        # Stop action doesn't have any navigation preceding it
        if last_action.is_stop():
            subgoal = TeachSubgoal.from_type_str_and_arg_id("Stop", -1)
            sample_out = copy.deepcopy(samples[start_idx])
            sample_out["subgoal"] = subgoal
            sample_out["eventual_action_ll"] = sample_out["action"]
            sample_out["eventual_action_observation"] = last_observation
            rollout_out.append(sample_out)

        else:
            assert self.is_sequence_terminal(last_action), (
                "Last action is high-level sequence should be either manipulation or stop")

            subgoal = TeachSubgoal.from_action_and_observation(last_action, last_observation)
            sample_out = copy.deepcopy(samples[start_idx])
            sample_out["subgoal"] = subgoal
            sample_out["eventual_action_ll"] = samples[-1]["action"]
            sample_out["eventual_action_observation"] = last_observation
            rollout_out.append(sample_out)
        return rollout_out
