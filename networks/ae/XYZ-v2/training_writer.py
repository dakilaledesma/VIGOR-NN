import json
from math import radians, cos


def convert(read, write, write2):
    array = open(read + '.json')
    save_file = open(write + '.csv', 'w')
    save_file2 = open(write2 + '.csv', 'w')
    userdata = json.load(array)

    for frame in range(0, 5000):
        for joint in [4, 5, 6]:
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
        for joint in [8, 9, 10]:
            save_file2.write(
                str(userdata["ModelQuaternionList"][frame]['V3BSVJoints'][int(joint)]['x'])
            )
            save_file2.write(",")
            save_file2.write(
                str(userdata["ModelQuaternionList"][frame]['V3BSVJoints'][int(joint)]['y'])
            )
            save_file2.write(",")
            save_file2.write(
                str(userdata["ModelQuaternionList"][frame]['V3BSVJoints'][int(joint)]['z'])
            )
            save_file2.write(",")
        save_file2.write("\n")
    save_file.close()
    save_file2.close()

convert('../JSONs/ae-naturalwalk-2', 'NW-CSVs/training-2-1', 'NW-CSVs/training-2-2')
