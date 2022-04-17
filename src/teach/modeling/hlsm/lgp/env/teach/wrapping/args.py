from lgp.env.teach.wrapping.paths import get_teach_root_path


def get_faux_args():
    teach_root_path = get_teach_root_path()
    args = type("FauxArgs",
                (object,),
                {
                    "reward_config": f"{teach_root_path}/models/config/rewards.json"
                })()
    return args
