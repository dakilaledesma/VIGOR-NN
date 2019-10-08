import json


def return_joint_cartesian(json_file):
    Motion_Structure = json.load(json_file)
    ModelDataList = Motion_Structure['ModelDataList']
    frames = []
    for frame, ModelData in enumerate(ModelDataList):
        joint_locations = []
        for JointData in ModelData["JointLocations"]:
            joint_locations.append(
                [JointData["location"]['x'], JointData["location"]['y'], JointData["location"]['z']])
        frames.append(joint_locations)

    return frames