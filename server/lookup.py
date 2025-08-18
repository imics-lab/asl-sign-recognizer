import json
from typing import Dict, Optional


def get_video_name(words_dict: Dict[str, list], word: str) -> Optional[str]:
    """
    Returns the video filename for the given word.
    The filename is derived as "<videoKey>.mp4" if present in the mapping.
    """
    if word not in words_dict:
        print(f"Warning: Word '{word}' not found in dictionary")
        return None

    if words_dict[word][1] is None:
        print(f"Warning: No video assigned for word '{word}'")
        return None

    return str(words_dict[word][1] + ".mp4")


def add_video_names_to_dict(words_dict: Dict[str, list], resources_dir: str) -> None:
    """
    Augments words_dict in place by filling the video keys from nslt_2000.json.
    resources_dir: path to the resources folder containing nslt_2000.json
    """
    try:
        with open(f"{resources_dir}/nslt_2000.json", 'r') as f:
            class_to_video_dict = json.load(f)
    except FileNotFoundError:
        print("Error: 'nslt_2000.json' not found in resources.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode JSON in 'nslt_2000.json'.")
        return

    for key in words_dict.keys():
        class_num = words_dict[key][0]
        matched = False

        for video_key, value in class_to_video_dict.items():
            try:
                if class_num == str(value["action"][0]):
                    words_dict[key][1] = video_key
                    matched = True
                    break
            except (KeyError, TypeError):
                print(f"Warning: Invalid entry in nslt_2000.json for {video_key}")

        if not matched:
            print(f"Warning: No video found for class {class_num} (word '{key}')")


def create_word_info_dict(resources_dir: str) -> Dict[str, list]:
    """
    Creates dict from wlasl_class_list.txt to dict {word: [classNum, videoKey]}
    resources_dir: path to the resources folder containing wlasl_class_list.txt
    """
    try:
        with open(f"{resources_dir}/wlasl_class_list.txt") as f:
            words_dict = {
                word: [num, None]
                for num, word in (line.strip().split('\t', 1) for line in f)
            }
    except FileNotFoundError:
        print("Error: 'wlasl_class_list.txt' not found in resources.")
        return {}

    add_video_names_to_dict(words_dict=words_dict, resources_dir=resources_dir)
    return words_dict


