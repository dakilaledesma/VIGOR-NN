import json
from math import radians, cos


def convert(read, write):
    array = open(read + '.json')
    save_file = open(write + '.csv', 'w')
    userdata = json.load(array)

    for frame in range(0, 5000):
        for joint in [12, 13, 14, 16, 17, 18]:
            save_file.write(
                str(userdata["ModelQuaternionList"][frame]['V3BSVJoints'][int(joint)]['x'])
            )
            save_file.write(",")
            save_file.write(
                str(userdata["ModelQuaternionList"][frame]['V3BSVJoints'][int(joint)]['y'])
            )
            save_file.write(",")
            save_file.write(
                str(userdata["ModelQuaternionList"][frame]['V3BSVJoints'][int(joint)]['z'])
            )
            save_file.write(",")
        save_file.write("\n")
    save_file.close()

convert('../JSONs/ae-naturalwalk-1', 'NW-CSVs/target-2')
