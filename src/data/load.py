import numpy as np


def load_iris(filepath):
    points = []
    labels = []
    name_dict = {}
    next_label = 0
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split(',')
            points.append([float(x) for x in parts[:-2]])
            name = parts[-1]
            if name in name_dict:
                labels.append(name_dict[name])
            else:
                labels.append(next_label)
                name_dict[name] = next_label
                next_label += 1

    return np.array(points, dtype=np.float32), labels, name_dict
