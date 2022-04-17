import os
import sys
sys.path.append('/home/ubuntu/efs/teach/src/teach/modeling/hlsm')
import torch
from lgp.parameters import Hyperparams
import lgp.paths
import pdb
import sys


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin



def build_teach_hierarchical_agent(agent_setup, hparams, device, exp_def=None):

    # Import agents (hierarchichal, high-level, and low-level)
    from lgp.agents.hierarchical_agent import HierarchicalAgent
    from lgp.agents.action_proposal_agent import ActionProposalAgent
    # Import model factory
    from lgp.models.teach.hlsm.hlsm_model_factory import HlsmModelFactory
    # Import classes
    from lgp.env.teach.teach_action import TeachAction
    from lgp.models.teach.hlsm.hlsm_task_repr import HlsmTaskRepr

    model_factory = HlsmModelFactory(hparams)
    skillset = model_factory.get_skillset(exp_def=exp_def)
    obsfunc = model_factory.get_observation_function()
    actprop = model_factory.get_subgoal_model()

    subgoal_model_path = lgp.paths.get_subgoal_model_path(exp_def=exp_def)
    if subgoal_model_path:
        sd = torch.load(subgoal_model_path)
        actprop.load_state_dict(sd, strict=False)

    actprop = actprop.to(device)

    obsfunc.eval()
    actprop.eval()

    highlevel_agent = ActionProposalAgent(actprop, obsfunc, HlsmTaskRepr, device)
    hierarchical_agent = HierarchicalAgent(highlevel_agent, skillset, obsfunc, TeachAction)
    return hierarchical_agent


def build_teach_deviant_agent(agent_setup, hparams, device, exp_def=None):
    deviance_p = hparams.get("agent_setup").get("deviance")
    from lgp.agents.deviant_agent import DeviantAgent
    from lgp.agents.teach.demonstration_replay_agent import DemonstrationReplayAgent
    from lgp.agents.teach.random_valid_agent import RandomValidAgent
    agent = DeviantAgent(oracle_agent=DemonstrationReplayAgent(),
                         random_agent=RandomValidAgent(),
                         deviance_prob=deviance_p)
    return agent


def build_demo_replay_agent(agent_setup, hparams, device, exp_def=None):
    from lgp.agents.teach.demonstration_replay_agent import DemonstrationReplayAgent
    agent = DemonstrationReplayAgent()
    return agent


def build_teach_random_agent(*args, **kwargs):
    from lgp.agents.teach.random_valid_agent import RandomValidAgent
    return RandomValidAgent()


AGENT_BUILDERS = {
    "teach_random_agent": build_teach_random_agent,
    "build_teach_hierarchical_agent": build_teach_hierarchical_agent,
    "teach_deviant_agent": build_teach_deviant_agent,
    "teach_demo_replay_agent": build_demo_replay_agent
}


def get_agent(setup: Hyperparams, hparams: Hyperparams, device=None, exp_def=None):

    agent_type = setup.agent_type
    agent_setup = setup.agent_setup
    if device is None:
        device = setup.device

    if agent_type not in AGENT_BUILDERS:
        raise ValueError(f"Unrecognized agent type: {agent_type}")

    return AGENT_BUILDERS[agent_type](agent_setup, hparams, device, exp_def=exp_def)
