import sys
sys.path.append("/home/ubuntu/efs/teach/src/teach/modeling/hlsm")
import pdb
from pathlib import Path
from typing import List
import argparse
import os
import json
import copy
import numpy as np
import torch
from collections import Counter, OrderedDict
from teach.modeling.hlsm.teach import constants
from teach.modeling.hlsm.teach.data.zoo.guides_edh import GuidesEdhDataset
from teach.modeling.hlsm.teach.data.preprocessor import Preprocessor
from teach.modeling.hlsm.teach.utils import data_util, eval_util, model_util
from teach.dataset.definitions import Definitions
from teach.inference.actions import obj_interaction_actions
from teach.inference.teach_model import TeachModel
from teach.logger import create_logger
from lgp.models.alfred.hlsm.alfred_perception_model import AlfredSegmentationAndDepthModel
from lgp.env.alfred.alfred_observation import AlfredObservation
from teach.modeling.hlsm.lgp.agents.agents import get_agent
import teach.modeling.hlsm.lgp.parameters as parameters
from lgp.models.teach.hlsm.hlsm_task_repr import HlsmTaskRepr
from lgp.rollout.rollout_actor import RolloutActorLocal
from lgp.env.alfred.alfred_env import AlfredEnv
from lgp import paths

from lgp.env.alfred.state_tracker import PoseInfo, InventoryInfo
from lgp.env.privileged_info import PrivilegedInfo

from teach.dataset.task_THOR import Task_THOR


logger = create_logger(__name__)

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

class HLSM_MODEL(TeachModel):
    """
    Wrapper around HLSM Model for inference
    """

    def __init__(self, process_index: int, num_processes: int, model_args: List[str]):
        """Constructor

        :param process_index: index of the eval process that launched the model
        :param num_processes: total number of processes launched
        :param model_args: extra CLI arguments to teach_eval will be passed along to the model
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=1, help="Random seed")
        parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
        parser.add_argument("--model_dir", type=str, required=True, help="Model folder name under $ET_LOGS")
        parser.add_argument("--checkpoint", type=str, default="latest.pth", help="latest.pth or model_**.pth")
        parser.add_argument("--subgoal_model_file", type=str, default="alfred_hlsm_subgoal_model_e5.pytorch", help="Path to subgoal model checkpoint")
        parser.add_argument("--depth_model_file", type=str, default="hlsm_depth_model_e3.pytorch", help="Path to depth model checkpoint")
        parser.add_argument("--seg_model_file", type=str, default="hlsm_segmentation_model_e4.pytorch", help="Path to segmentation model checkpoint")
        parser.add_argument("--navigation_model_file", type=str, default="hlsm_gofor_navigation_model_e5.pytorch", help="Path to navigation model checkpoint")
        parser.add_argument("--experirment_def_name", type=str, default="", help="Name of the experiment to run")
        parser.add_argument(
            "--skip_edh_history",
            action="store_true",
            default=False,
            help="Specify this to ignore actions and image frames in EDH history",
        )

        args = parser.parse_args(model_args)
        args.dout = args.model_dir
        self.args = args

        logger.info("HLSM using args %s" % str(args))
        np.random.seed(args.seed)

        self.object_predictor = None
        self.model = None
        self.extractor = None
        self.vocab = None
        self.preprocessor = None

        self.input_dict = None
        self.cur_edh_instance = None
        #ForkedPdb().set_trace()
        self.exp_def = parameters.Hyperparams(parameters.load_experiment_definition(self.args.experirment_def_name))
        parameters.EXPERIMENT_DEFINITION = self.exp_def
        self.set_up_model(process_index)
        device = torch.device(self.exp_def.Setup.device)
        self.latest_extra_events = []
        self.smooth_nav = self.exp_def.Setup.env_setup.d.get("smooth_nav")
        self.fail_count = 0
        self.max_fails = 10
        self.counter = 0

        # self.env = AlfredEnv(device=device,
        #             setup=self.exp_def.Setup.env_setup.d,
        #             hparams=self.exp_def.Hyperparams.d)
        # from teach.settings import get_settings
        # from teach.simulators.simulator_THOR import COMMIT_ID, TEAChController
        #self.env = TEAChController(base_dir=get_settings().AI2THOR_BASE_DIR, download_only=True, commit_id=COMMIT_ID)



    # def load_checkpoint(self, checkpoint_file, model_file):
    #     model_state = torch.load(model_file)
    #     checkpoint = torch.load(checkpoint_file)
    #     nonbert_optimizer_state = checkpoint["nonbert_optimizer"]
    #     bert_optimizer_state = checkpoint["bert_optimizer"]
    #     epoch = checkpoint["epoch"]
    #     iter = checkpoint["iter"]
    #     return (nonbert_optimizer_state, bert_optimizer_state), model_state, epoch, iter

    def set_up_model(self, process_index):

        os.makedirs(self.args.dout, exist_ok=True)
        model_path = os.path.join(self.args.model_dir, self.args.checkpoint)
        logger.info("Loading model from %s" % model_path)
        logger.info(f"Loading model agent using device: {self.args.device}")
        agent = get_agent(self.exp_def.Setup, self.exp_def.Hyperparams, self.args.device, self.exp_def)
        self.model = agent
        self.PERCEPTION_DEVICE = "cuda"
        self.depth_model = AlfredSegmentationAndDepthModel(self.exp_def.Hyperparams).to(self.PERCEPTION_DEVICE)
        depth_model_path = self.exp_def.Setup.agent_setup.depth_model_file
        depth_model_path = os.path.join(paths.MODEL_PATH, depth_model_path)
        self.depth_model.load_state_dict(torch.load(depth_model_path))
        self.depth_model.eval()
        self.seg_model = AlfredSegmentationAndDepthModel(self.exp_def.Hyperparams).to(self.PERCEPTION_DEVICE)
        seg_model_path = self.exp_def.Setup.agent_setup.seg_model_file
        seg_model_path = os.path.join(paths.MODEL_PATH, seg_model_path)
        self.seg_model.load_state_dict(torch.load(seg_model_path))
        self.seg_model.eval()
        self.latest_observation = None
        self.latest_action = None
        self.fov = 60
        self.steps = 0
        self.device = None
        return agent

    def get_rollout_actor(self, ):
        self.rollout_actor = RolloutActorLocal(experiment_name=exp_name,
                                      agent=self.agent,
                                      env=env,
                                      dataset_proc=None,
                                      param_server_proc=None,
                                      max_horizon=horizon,
                                      dataset_device=dataset_device,
                                      index=1,
                                      collect_trace=visualize_rollouts,
                                      lightweight_mode=not visualize_rollouts)

    def _extract_reference_semantic_image(self, event):
        num_objects = segdef.get_num_objects()
        h, w = event.frame.shape[0:2]
        seg_image = torch.zeros([num_objects, h, w], dtype=torch.int16, device=device)

        inventory_obj_strs = set()
        for object in event.metadata['inventoryObjects']:
            inventory_obj_string = object['objectType'].split("_")[0]
            inventory_obj_strs.add(inventory_obj_string)

        for obj_str, class_mask in event.class_masks.items():
            obj_int = segdef.object_string_to_intid(obj_str)
            class_mask_t = torch.from_numpy(class_mask.astype(np.int16)).to(device)
            seg_image[obj_int] = torch.max(seg_image[obj_int], class_mask_t)
        return seg_image.type(torch.ByteTensor)

    def _make_observation(self):

        event = self.latest_event
        if event.frame is not None:
            rgb_image = torch.from_numpy(event.frame.copy()).permute((2, 0, 1)).unsqueeze(0).half() / 255
        else:
            rgb_image = torch.zeros((1, 3, 300, 300))
        # Depth
        if self.exp_def.Setup.env_setup.d.get("reference_depth"):
            depth_image = torch.from_numpy(event.depth_frame.copy()).unsqueeze(0).unsqueeze(0) / 1000
        else:
            _, pred_depth = self.depth_model.predict(rgb_image.float().to(self.PERCEPTION_DEVICE))
            depth_image = pred_depth.to("cpu") # TODO: Maybe skip this? We later move it to GPU anyway

        # Segmentation
        if self.exp_def.Setup.env_setup.d.get("reference_segmentation"):
            semantic_image = self._extract_reference_semantic_image(event)
            semantic_image = semantic_image.unsqueeze(0)
        else:
            pred_seg, _ = self.seg_model.predict(rgb_image.float().to(self.PERCEPTION_DEVICE))
            semantic_image = pred_seg

                # Simple error detection from RGB image changes
        action_failed = False
        if self.latest_observation is not None:
            assert self.latest_action is not None, "Didn't log an action, but got two observations in a row?"
            rgb_diff = (rgb_image - self.latest_observation.rgb_image).float().abs().mean()
            if rgb_diff < 1e-4:
                print(f"Action: {self.latest_action}, RGB Diff: {rgb_diff}. Counting as failed.")
                action_failed = True
            else:
                pass

        if not action_failed and self.latest_action is not None:
            self.pose_info.simulate_successful_action(self.latest_action)
            oinv = copy.deepcopy(self.inventory_info)
            self.inventory_info.simulate_successful_action(self.latest_action, self.latest_observation)
            if len(oinv.inventory_object_ids) != len(self.inventory_info.inventory_object_ids):
                print(self.inventory_info.summarize())

        # Pose
        if self.exp_def.Setup.env_setup.d.get("reference_pose"):
            self.pose_info = PoseInfo.from_ai2thor_event(event)

        T_world_to_cam = self.pose_info.get_pose_mat()
        cam_horizon_deg = [self.pose_info.cam_horizon_deg]
        agent_pos = self.pose_info.get_agent_pos()

        # Inventory
        if self.exp_def.Setup.env_setup.d.get("reference_inventory"):
            self.inventory_info = InventoryInfo.from_ai2thor_event(event)
        inventory_vector = self.inventory_info.get_inventory_vector()
        inventory_vector = inventory_vector.unsqueeze(0)

        privileged_info = PrivilegedInfo(event)

        observation = AlfredObservation(rgb_image,
                                         depth_image,
                                         semantic_image,
                                         inventory_vector,
                                         T_world_to_cam,
                                         self.fov,
                                         cam_horizon_deg,
                                         privileged_info)
        observation.set_agent_pos(agent_pos)
        if action_failed:
            observation.set_error_causing_action(self.latest_action)

        # Add extra RGB frames from smooth navigation
        if self.latest_extra_events:
            extra_frames = [torch.from_numpy(e.frame.copy()).permute((2, 0, 1)).unsqueeze(0).half() / 255 for e in self.latest_extra_events]
            observation.extra_rgb_frames = extra_frames
        task = None

        return observation


    def reset(self, event):
        # First reset everything
        self.latest_event = event
        self.first_event = event
        self.latest_action = None
        self.latest_observation = None

        # Initialize pose and inventory
        if self.exp_def.Setup.env_setup.d.get("reference_pose"):
            self.pose_info = PoseInfo.from_ai2thor_event(event)
        else:
            self.pose_info = PoseInfo.create_new_initial()

        if self.exp_def.Setup.env_setup.d.get("reference_inventory"):
            self.inventory_info = InventoryInfo.from_ai2thor_event(event)
        else:
            self.inventory_info = InventoryInfo.create_empty_initial()

        # Make the first observation
        self.latest_observation = copy.deepcopy(self._make_observation())
        return self.latest_observation


    def start_new_edh_instance(self, original_edh_instance, edh_instance, edh_history_images, simulator, task, instance_file, game_file, edh_name=None):

        game = json.load(open(game_file, "r"))
        event = simulator.controller.last_event
        observation = self.reset(event)
        self.model.start_new_rollout(task)
        self.task = task
        self.game = game


        return True


    def to_thor_api_exec(self, action, simulator, object_id="", smooth_nav=False, debug_print_all_sim_steps=True):
        # TODO: parametrized navigation commands
        print(action)
        if action in ["Forward", "MoveAhead", "Move Ahead"]:
            ac = dict(action="MoveAhead", forceAction=True)
            if debug_print_all_sim_steps:
                logger.info("step %s", ac)
            e = simulator.controller.step(ac)
        elif action in ["Backward", "MoveBack", "Move Back"]:
            ac = dict(action="MoveBack", forceAction=True)
            if debug_print_all_sim_steps:
                logger.info("step %s", ac)
            e = simulator.controller.step(ac)
        elif action in ["Look Up", "LookUp"]:
            ac = dict(action="LookUp", forceAction=True)
            if debug_print_all_sim_steps:
                logger.info("step %s", ac)
            e = simulator.controller.step(ac)
        elif action in ["Look Down", "LookDown"]:
            ac = dict(action="LookDown", forceAction=True)
            if debug_print_all_sim_steps:
                logger.info("step %s", ac)
            e = simulator.controller.step(ac)
        elif action in ["Turn Left", "TurnLeft", "RotateLeft", "Rotate Left"]:
            ac = dict(action="RotateLeft", forceAction=True)
            if debug_print_all_sim_steps:
                logger.info("step %s", ac)
            e = simulator.controller.step(ac)
        elif action in ["Turn Right", "TurnRight", "RotateRight", "Rotate Right"]:
            ac = dict(action="RotateRight", forceAction=True)
            if debug_print_all_sim_steps:
                logger.info("step %s", ac)
            e = simulator.controller.step(ac)
        elif action in ["Pan Left", "PanLeft", "MoveLeft", "Move Left"]:  # strafe left
            ac = dict(action="MoveLeft", forceAction=True)
            if debug_print_all_sim_steps:
                logger.info("step %s", ac)
            e = simulator.controller.step(ac)
        elif action in ["Pan Right", "PanRight", "MoveRight", "Move Right"]:  # strafe right
            ac = dict(action="MoveRight", forceAction=True)
            if debug_print_all_sim_steps:
                logger.info("step %s", ac)
            e = simulator.controller.step(ac)
        elif action == "Stop":  # do nothing
            ac = dict(action="Pass")
            if debug_print_all_sim_steps:
                logger.info("step %s", ac)
            e = simulator.controller.step(ac)
        else:
            logger.warning("%s: Motion not supported" % action)
            interaction.action.success = 0
            return False, "", None

        return e, action


    def _error_is_fatal(self, err):
        self.fail_count += 1
        if self.fail_count >= self.max_fails:
            print(f"EXCEEDED MAXIMUM NUMBER OF FAILURES ({self.max_fails})")
            return True
        else:
            return False

    def prune_by_any_interaction(self, simulator, instances_ids):
        '''
        ignores any object that is not interactable in anyway
        '''
        pruned_instance_ids = []
        for obj in simulator.controller.last_event.metadata['objects']:
            obj_id = obj['objectId']
            if obj_id in instances_ids:
                if obj['pickupable'] or obj['receptacle'] or obj['openable'] or obj['toggleable'] or obj['sliceable']:
                    pruned_instance_ids.append(obj_id)

        ordered_instance_ids = [id for id in instances_ids if id in pruned_instance_ids]
        return ordered_instance_ids

    def va_interact(self, action, simulator, interact_mask=None, smooth_nav=True, mask_px_sample=1, debug=False):
        '''
        interact mask based action call
        '''


        all_ids = []

        if type(interact_mask) is str and interact_mask == "NULL":
            raise Exception("NULL mask.")
        elif interact_mask is not None:
            # ground-truth instance segmentation mask from THOR
            instance_segs = np.array(simulator.controller.last_event.instance_segmentation_frame)

            if debug:
                print("step %s", dict(action="Pass", renderObjectImage=True))
            if instance_segs is None:
                simulator.controller.step(action="Pass", renderObjectImage=True)
                instance_segs = np.array(simulator.controller.last_event.instance_segmentation_frame)

            color_to_object_id = simulator.controller.last_event.color_to_object_id

            # get object_id for each 1-pixel in the interact_mask
            nz_rows, nz_cols = np.nonzero(interact_mask)
            instance_counter = Counter()
            for i in range(0, len(nz_rows), mask_px_sample):
                x, y = nz_rows[i], nz_cols[i]
                #ForkedPdb().set_trace()
                instance = tuple(instance_segs[x, y])
                instance_counter[instance] += 1
            if debug:
                print("action_box", "instance_counter", instance_counter)

            # iou scores for all instances
            iou_scores = {}
            for color_id, intersection_count in instance_counter.most_common():
                union_count = np.sum(np.logical_or(np.all(instance_segs == color_id, axis=2), interact_mask.astype(bool)))
                iou_scores[color_id] = intersection_count / float(union_count)
            iou_sorted_instance_ids = list(OrderedDict(sorted(iou_scores.items(), key=lambda x: x[1], reverse=True)))

            # get the most common object ids ignoring the object-in-hand
            inv_obj = simulator.controller.last_event.metadata['inventoryObjects'][0]['objectId'] \
                if len(simulator.controller.last_event.metadata['inventoryObjects']) > 0 else None
            all_ids = [color_to_object_id[color_id] for color_id in iou_sorted_instance_ids
                       if color_id in color_to_object_id and color_to_object_id[color_id] != inv_obj]

            # print all ids
            if debug:
                print("action_box", "all_ids", all_ids)

            # print instance_ids
            instance_ids = [inst_id for inst_id in all_ids if inst_id is not None]
            if debug:
                print("action_box", "instance_ids", instance_ids)
            # prune invalid instances like floors, walls, etc.
            instance_ids = self.prune_by_any_interaction(simulator, instance_ids)

            # cv2 imshows to show image, segmentation mask, interact mask
            if debug:
                print("action_box", "instance_ids", instance_ids)
                instance_seg = copy.copy(instance_segs)
                instance_seg[:, :, :] = interact_mask[:, :, np.newaxis] == 1
                instance_seg *= 255

                cv2.imshow('seg', instance_segs)
                cv2.imshow('mask', instance_seg)
                cv2.imshow('full', simulator.controller.last_event.frame[:,:,::-1])
                cv2.waitKey(0)

            if len(instance_ids) == 0:
                err = "Bad interact mask. Couldn't locate target object"
                success = False
                return success, None, None, err, None

            target_instance_id = instance_ids[0]
        else:
            target_instance_id = ""

        if debug:
            print("taking action: " + str(action) + " on target_instance_id " + str(target_instance_id))
        #ForkedPdb().set_trace()
        # event, api_action = self.to_thor_api_exec(action, simulator, target_instance_id, smooth_nav)
        try:
            event, api_action = self.to_thor_api_exec(action, simulator, target_instance_id, smooth_nav)
        except Exception as err:
            success = False
            return success, None, None, err, None

        if not event.metadata['lastActionSuccess']:
            if interact_mask is not None and debug:
                print("Failed to execute action!", action, target_instance_id)
                print("all_ids inside BBox: " + str(all_ids))
                instance_seg = copy.copy(instance_segs)
                instance_seg[:, :, :] = interact_mask[:, :, np.newaxis] == 1
                cv2.imshow('seg', instance_segs)
                cv2.imshow('mask', instance_seg)
                cv2.imshow('full', simulator.controller.last_event.frame[:,:,::-1])
                cv2.waitKey(0)
                print(event.metadata['errorMessage'])
            success = False
            return success, event, target_instance_id, event.metadata['errorMessage'], api_action

        success = True
        return success, event, target_instance_id, '', api_action


    def step(self, action, simulator):

        # The ALFRED API does not accept the Stop action, do nothing
        message = ""
        #ForkedPdb().set_trace()
        if action.is_stop():
            done = True
            transition_reward = 0
            api_action = None
            events = []

        # Execute all other actions in the ALFRED API
        else:
            definitions = Definitions(version="2.0")
            #action_definition = definitions.map_actions_id2info[action.action_id]
            alfred_action, interact_mask = action.to_teach_api()

            ret = self.va_interact(alfred_action, simulator, interact_mask, smooth_nav=self.smooth_nav)

            # Default version of ALFRED
            if len(ret) == 5:
                exec_success, event, target_instance_id, err, api_action = ret
                events = []
            # Patched version of ALFRED that returns intermediate events from smooth actions
            # To use this, apply the patch alfred-patch.patch onto the ALFRED code:
            # $ git am alfred-patch.patch
            elif len(ret) == 6:
                exec_success, event, events, target_instance_id, err, api_action = ret
            else:
                raise ValueError("Invalid number of return values from ThorEnv")
            #ForkedPdb().set_trace()

            # if not self.task.traj_data.is_test():
            #     transition_reward, done = self.thor_env.get_transition_reward()
            #     done = False
            # else:
            transition_reward, done = 0, False

            if not exec_success:
                fatal = self._error_is_fatal(err)
                print(f"ThorEnv {'fatal' if fatal else 'non-fatal'} Exec Error: {err}")
                if fatal:
                    done = True
                    api_action = None
                message = str(err)

        #self.prof.tick("step")

        # Track state (pose and inventory) from RGB images and actions
        event = simulator.controller.last_event
        # self.state_tracker.log_action(action)
        self.latest_action = action
        self.latest_event = event
        self.latest_extra_events = events
        self.latest_observation = copy.deepcopy(self._make_observation())

        observation = copy.deepcopy(self.latest_observation)
        observation.privileged_info.attach_task(self.task) # TODO: See if we can get rid of this?
        if self.device:
            observation = observation.to(self.device)

        # if not self.task.traj_data.is_test():
        #     reward = transition_reward - 0.05
        #     goal_satisfied = self.thor_env.get_goal_satisfied()
        #     goal_conditions_met = self.thor_env.get_goal_conditions_met()
        #     task_success = goal_satisfied
        #     md = {
        #         "success": task_success,
        #         "goal_satisfied": goal_satisfied,
        #         "goal_conditions_met": goal_conditions_met,
        #         "message": message,
        #     }
        # else:
        reward = 0
        md = {}

        self.steps += 1

        return observation, reward, done, md, exec_success

    def get_next_action(self, img, original_edh_instance, edh_instance, prev_action, simulator, img_name=None, edh_name=None):
        """
        Sample function producing random actions at every time step. When running model inference, a model should be
        called in this function instead.
        :param img: PIL Image containing agent's egocentric image
        :param edh_instance: EDH instance
        :param prev_action: One of None or a dict with keys 'action' and 'obj_relative_coord' containing returned values
        from a previous call of get_next_action
        :param img_name: image file name
        :param edh_name: EDH instance file name
        :return action: An action name from all_agent_actions
        :return obj_relative_coord: A relative (x, y) coordinate (values between 0 and 1) indicating an object in the image;
        The TEACh wrapper on AI2-THOR examines the ground truth segmentation mask of the agent's egocentric image, selects
        an object in a 10x10 pixel patch around the pixel indicated by the coordinate if the desired action can be
        performed on it, and executes the action in AI2-THOR.
        """
        # img_feat = self.extractor.featurize([img], batch=1)
        # self.input_dict["frames"] = img_feat

        #ForkedPdb().set_trace()
        with torch.no_grad():
            #prev_api_action = None
            #if prev_action is not None and "action" in prev_action:
            #    prev_api_action = prev_action["action"]

            next_observation, reward, done, md, exec_success = self.step(prev_action, simulator)
            #ForkedPdb().set_trace()
            action = self.model.act(next_observation)
            #ForkedPdb().set_trace()
        self.counter += 1
        return action, exec_success#predicted_click

    def get_obj_click(self, obj_class_idx, img):
        rcnn_pred = self.object_predictor.predict_objects(img)
        obj_class_name = self.object_predictor.vocab_obj.index2word(obj_class_idx)
        candidates = list(filter(lambda p: p.label == obj_class_name, rcnn_pred))
        if len(candidates) == 0:
            return [np.random.uniform(), np.random.uniform()]
        index = np.argmax([p.score for p in candidates])
        mask = candidates[index].mask[0]
        predicted_click = list(np.array(mask.nonzero()).mean(axis=1))
        predicted_click = [
            predicted_click[0] / mask.shape[1],
            predicted_click[1] / mask.shape[0],
        ]
        return predicted_click

    def obstruction_detection(self, action, prev_action_success, m_out, vocab_out):
        """
        change 'MoveAhead' action to a turn in case if it has failed previously
        """
        if action != "Forward" or prev_action_success:
            return action
        dist_action = m_out["action"][0][0].detach().cpu()
        idx_rotateR = vocab_out.word2index("Turn Right")
        idx_rotateL = vocab_out.word2index("Turn Left")
        action = "Turn Left" if dist_action[idx_rotateL] > dist_action[idx_rotateR] else "Turn Right"
        logger.debug("Blocking action is changed to: %s" % str(action))
        return action
