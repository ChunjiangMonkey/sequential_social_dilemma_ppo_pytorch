import numpy as np


def create_dict_with_same_keys(original_dict):
    new_dict = {}
    for key, value in original_dict.items():
        if isinstance(value, dict):
            new_dict[key] = create_dict_with_same_keys(value)
        else:
            new_dict[key] = []
    return new_dict


def reset_dict(original_dict):
    for key, value in original_dict.items():
        if isinstance(value, dict):
            reset_dict(value)
        else:
            original_dict[key] = []


def info_dict_list_to_mean_info_dict(infos_dict_list):
    # convert the dict list (such as [{"a":0,"b":1},{"a":1,"b":2}]) to mean dict (such as {"a":0.5,"b":1.5})
    infos_dict = create_dict_with_same_keys(infos_dict_list[0])
    for key in infos_dict.keys():
        if isinstance(infos_dict[key], dict):
            for sub_key in infos_dict[key].keys():
                infos_dict[key][sub_key] = np.mean([info[key][sub_key] for info in infos_dict_list])
        else:
            infos_dict[key] = np.mean([info[key] for info in infos_dict_list])
    return infos_dict


# def info_dict_list_to_mean_info_dict(dict_list):
#     # convert the dict list (such as [{"a":0,"b":1},{"a":1,"b":2}]) to mean dict (such as {"a":0.5,"b":1.5})
#     if not dict_list:
#         return {}
#     elif all(isinstance(item, dict) for item in dict_list):
#         mean_dict = {}
#         for key in dict_list[0]:
#             values = [d[key] for d in dict_list if key in d]
#             mean_dict[key] = info_dict_list_to_mean_info_dict(values)
#         return mean_dict
#     else:
#         return np.mean(dict_list)


def info_list_dict_to_mean_info_dict(infos_list_dict):
    # convert the list dict (such as {"a":[0,1],"b":[1,2]}) to mean dict (such as {"a":0.5,"b":1.5})
    infos_dict = create_dict_with_same_keys(infos_list_dict)
    for key in infos_dict.keys():
        if isinstance(infos_dict[key], dict):
            for sub_key in infos_dict[key].keys():
                infos_dict[key][sub_key] = np.mean(infos_list_dict[key][sub_key])
        else:
            infos_dict[key] = np.mean(infos_list_dict[key])
    return infos_dict
