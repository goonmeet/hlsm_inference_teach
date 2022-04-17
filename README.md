# TEACh
[Task-driven Embodied Agents that Chat](https://arxiv.org/abs/2110.00534)

Aishwarya Padmakumar*, Jesse Thomason*, Ayush Shrivastava, Patrick Lange, Anjali Narayan-Chen, Spandana Gella, Robinson Piramuthu, Gokhan Tur, Dilek Hakkani-Tur

TEACh is a dataset of human-human interactive dialogues to complete tasks in a simulated household environment.

## Prerequisites
- python3 `>=3.7,<=3.8`
- python3.x-dev, example: `sudo apt install python3.8-dev`
- tmux, example: `sudo apt install tmux`
- xorg, example: `sudo apt install xorg openbox`
- ffmpeg, example: `sudo apt install ffmpeg`

## Installation
```
pip install -r requirements.txt
pip install -e .
```
## Downloading the dataset
Run the following script:
```
teach_download
```
This will download and extract the archive files (`experiment_games.tar.gz`, `all_games.tar.gz`,
`images_and_states.tar.gz`, `edh_instances.tar.gz` & `tfd_instances.tar.gz`) in the default
directory (`/tmp/teach-dataset`).  
**Optional arguments:**
- `-d`/`directory`: The location to store the dataset into. Default=`/tmp/teach-dataset`.
- `-se`/`--skip-extract`: If set, skip extracting archive files.
- `-sd`/`--skip-download`: If set, skip downloading archive files.
- `-f`/`--file`: Specify the file name to be retrieved from S3 bucket.

File changes (12/28/2022):
We have modified EDH instances so that the state changes checked for to evaluate success are only those that contribute towards task success in the main task of the gameplay session the EDH instance is created from.
We have removed EDH instances that had no state changes meeting these requirements.
Additionally, two game files, and their corresponding EDH and TfD instances were deleted from the `valid_unseen` split due to issues in the game files.
Version 3 of our paper on Arxiv, which will be public on Dec 30, 2022 contains the updated dataset size and experimental results.  

## Remote Server Setup
If running on a remote server without a display, the following setup will be needed to run episode replay, model inference of any model training that invokes the simulator (student forcing / RL).

Start an X-server
```
tmux
sudo python ./bin/startx.py
```
Exit the `tmux` session (`CTRL+B, D`). Any other commands should be run in the main terminal / different sessions.

## Data Setup
Download Teach Data

Get Teach Data that is formatted for Alfred from Vardaan

For this setup the dataset directory is outside of this directory (i.e, ../teach-dataset)

For this setup the dataset directory is outside of this directory (i.e, ../hlsm_formatted_edh_format)

For later: reduce the dataset redundancy between  ../teach-dataset and ../hlsm_formatted_edh_format

### Running without docker

#### Inference
`source init.sh`

`source env_setup.sh`

`source venv/teach-hlsm/bin/activate`

`bash hlsm_teach_without_docker.sh`

The evaluation metrics will be in `$HOST_OUTPUT_DIR/$SUBMISSION_PK/metrics_file`.
Images for each episode will be in `$HOST_IMAGES_DIR/$SUBMISSION_PK`.
