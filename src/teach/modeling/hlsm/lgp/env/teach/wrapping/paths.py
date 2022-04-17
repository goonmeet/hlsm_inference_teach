from typing import List
import os, pdb


def get_teach_root_path():
    assert 'ALFRED_ROOT' in os.environ, "ALFRED_ROOT environment variable not defined!"
    return os.environ['ALFRED_ROOT']


def get_task_traj_data_path(data_split: str, task_id: str) -> str:
    traj_data_path = os.path.join(os.environ['ET_DATA'], "edh_instances", data_split, task_id)
    return traj_data_path


def get_traj_data_paths(data_split: str) -> List[str]:
    teach_root = get_teach_root_path()
    traj_data_root = os.path.join(teach_root, "data", "json_2.1.0", data_split)
    all_tasks = os.listdir(traj_data_root)
    traj_data_paths = []
    for task in all_tasks:
        trials = os.listdir(os.path.join(traj_data_root, task))
        for trial in trials:
            traj_data_paths.append(os.path.join(traj_data_root, task, trial, "traj_data.json"))

    if (input_path / "processed.txt").exists():
        # the dataset was generated locally
        with (input_path / "processed.txt").open() as f:
            traj_paths = [line.strip() for line in f.readlines()]
            traj_paths = [line.split(";")[0] for line in traj_paths if line.split(";")[1] == "1"]
            traj_paths = [str(input_path / line) for line in traj_paths]
    else:
        # the dataset was downloaded from ALFRED servers
        traj_paths_all = sorted([str(path) for path in input_path.glob("*/*.json")])
        traj_paths = traj_paths_all
    if fast_epoch:
        traj_paths = traj_paths[::20]
    num_files = len(traj_paths)
    if processed_files_path is not None and processed_files_path.exists():
        if str(processed_files_path).endswith(constants.VOCAB_FILENAME):
            traj_paths = []
        else:
            with processed_files_path.open() as f:
                processed_files = set([line.strip() for line in f.readlines()])
            traj_paths = [traj for traj in traj_paths if traj not in processed_files]
    traj_paths = [Path(path) for path in traj_paths]
    return traj_data_paths


def get_task_dir_path(data_split: str, task_id: str) -> str:
    teach_root = get_teach_root_path()
    task_dir_path = os.path.join(teach_root, "data", "json_2.1.0", data_split, task_id.split("/")[0])
    return task_dir_path


def get_splits_path():

    splits_path = os.listdir(os.path.join(os.environ['ET_DATA'], "edh_instances"))
    splits_to_paths = {}
    for split in splits_path:
        splits_to_paths[split] = []
        instances_splits_files = os.listdir(os.path.join(os.environ['ET_DATA'], "edh_instances", split))
        for file in instances_splits_files:
            data = {'repeat_idx': 0, 'task': file}
            splits_to_paths[split].append(data)
    return splits_to_paths
