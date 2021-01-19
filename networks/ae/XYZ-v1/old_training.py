import json
from math import radians, cos


def convert(read, write):
    array = open(read + '.json')
    save_file = open(write + '.csv', 'w')
    userdata = json.load(array)

    for frame in range(0, 5000):
        for joint in "0123":
            if joint in "45":
                save_file.write("0.0,0.0,0.0,")
            else:
                save_file.write(
                    str(transform(userdata["ModelQuaternionList"][frame]['ChildQuaternionList'][int(joint)]['euler']['x']))
                )
                save_file.write(",")
                save_file.write(
                    str(transform(userdata["ModelQuaternionList"][frame]['ChildQuaternionList'][int(joint)]['euler']['y']))
                    )
                save_file.write(",")
                save_file.write(
                    str(transform(userdata["ModelQuaternionList"][frame]['ChildQuaternionList'][int(joint)]['euler']['z']))
                    )
                save_file.write(",")
        save_file.write("\n")
    save_file.close()


def transform(data):
    halved_data = data/2
    radians_data = radians(halved_data)
    if data > 180:
        cos_data = cos(radians_data) * -1
    else:
        cos_data = cos(radians_data)
    return cos_data


convert('JSONs/ae-naturalwalk-1', 'training-transformed')
