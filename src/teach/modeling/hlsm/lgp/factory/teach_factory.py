import sys
sys.path.append('/home/ubuntu/efs/teach/src/teach/modeling/hlsm/workspace/alfred_src/')

from typing import Dict
from lgp.abcd.factory import Factory

from lgp.env.teach.teach_env import TeachEnv
from lgp.agents.agents import get_agent
from lgp.models.teach.hlsm.hlsm_model_factory import HlsmModelFactory

from lgp.parameters import Hyperparams


class TeachFactory(Factory):
    def __init__(self):
        super().__init__()

    def get_model_factory(self, setup: Dict, hparams : Hyperparams):
        # TODO support picking between multiple models if needed
        return HlsmModelFactory(hparams)

    def get_environment(self, setup : Dict, task_num_range=None):
        # TODO: Retrieve train/dev/test split based on setup
        device = setup.get("device", "cpu")
        env = TeachEnv(device=device, setup=setup["env_setup"])
        env.set_task_num_range(task_num_range)
        return env

    def get_agent(self, setup : Hyperparams, hparams : Hyperparams):
        return get_agent(setup, hparams)
