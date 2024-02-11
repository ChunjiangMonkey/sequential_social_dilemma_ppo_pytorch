import numpy as np


def create_dict_with_same_keys(original_dict):
    new_dict = {}
    for key, value in original_dict.items():
        if isinstance(value, dict):
            new_dict[key] = create_dict_with_same_keys(value)
        else:
            new_dict[key] = []
    return new_dict


def to_mean_info_dict(infos_list):
    infos_dict = create_dict_with_same_keys(infos_list[0])
    for key in infos_dict.keys():
        if isinstance(infos_dict[key], dict):
            for sub_key in infos_dict[key].keys():
                infos_dict[key][sub_key] = np.mean([info[key][sub_key] for info in infos_list])
        else:
            infos_dict[key] = np.mean([info[key] for info in infos_list])
    return infos_dict



