import json

def get_video_name(wordsDict, word):
    """
    Returns video filename for passed word.
    """
    if word not in wordsDict:
        print(f"Warning: Word '{word}' not found in dictionary")
        return None

    if wordsDict[word][1] is None:
        print(f"Warning: No video assigned for word '{word}'")
        return None

    return str(wordsDict[word][1] + ".mp4")

def add_video_names_to_dict(wordsDict):
    """
    Adds video file names to dict.
    """
    try:
        with open('resources/nslt_2000.json', 'r') as f:
            classToVideoDict = json.load(f)
    except FileNotFoundError:
        print("Error: 'resources/nslt_2000.json' not found.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode JSON in 'resources/nslt_2000.json'.")
        return

    for key in wordsDict.keys():
        classNum = wordsDict[key][0]
        matched = False

        # Finds matching classNum in classToVideoDict
        for videoKey, value in classToVideoDict.items():
            try:
                if classNum == str(value["action"][0]):
                    wordsDict[key][1] = videoKey
                    matched = True
                    break
            except (KeyError, TypeError):
                print(f"Warning: Invalid entry in nslt_2000.json for {videoKey}")

        if not matched:
            print(f"Warning: No video found for class {classNum} (word '{key}')")

def create_word_info_dict():
    """
    Creates dict from wlasl_class_list.txt to dict {word: [classNum, videoName]}
    """

    try:
        with open('resources/wlasl_class_list.txt') as f:
            wordsDict = {
                # Strips and splits each line and rearranges from (classNum: word) to (word: classNum)
                word: [num, None] for num, word in (line.strip().split('\t', 1) for line in f)
            }
    except FileNotFoundError:
        print("Error: 'resources/wlasl_class_list.txt' not found.")
        return {}
    
    add_video_names_to_dict(wordsDict=wordsDict)
    return wordsDict