from itertools import product

letters = "GBRY"
nums = "1234"
levels = [ch + num for num, ch in product(nums, letters)]
level_codes = [2 ** i for i in range(len(levels))]
code_to_level = {i: j for i, j in zip(level_codes, levels)}
level_to_code = {j: i for i, j in zip(level_codes, levels)}

def read_seg(filename: str, encoding: str = "utf-8-sig") -> tuple[dict, list[dict]]:
    with open(filename, encoding=encoding) as f:
        lines = [line.strip() for line in f.readlines()]

    header_start = lines.index("[PARAMETERS]") + 1
    data_start = lines.index("[LABELS]") + 1

    params = {}
    for line in lines[header_start:data_start - 1]:
        key, value = line.split("=")
        params[key] = int(value)

    labels = []
    for line in lines[data_start:]:
        if line.count(",") < 2:
            break
        pos, level, name = line.split(",", maxsplit=2)
        label = {
            "position": int(pos) // params["BYTE_PER_SAMPLE"] // params["N_CHANNEL"],
            "level": code_to_level[int(level)],
            "name": name
        }
        labels.append(label)
    return params, labels

def write_seg_light(params, labels, filename, encoding = "utf-8-sig"):
    params["N_LABEL"] = len(labels)
    with open(filename, "w", encoding=encoding) as f:
        f.write("[PARAMETERS]\n")
        for key, value in params.items():
            f.write(f"{key}={value}\n")
        f.write("[LABELS]\n")
        for label in labels:
            f.write(f"{params['BYTE_PER_SAMPLE'] * params['N_CHANNEL'] * label['position']},")
            f.write(f"{level_to_code[label['level']]},")
            f.write(f"{label['name']}\n")

def write_seg_with_params(s_freq, labels, names, filename, encoding = "utf-8-sig"):
    param_defaults = {
        "SAMPLING_FREQ": s_freq,
        "BYTE_PER_SAMPLE": 2,
        "CODE": 0,
        "N_CHANNEL": 1,
        "N_LABEL": 0
    }
    param_defaults["N_LABEL"] = len(labels)
    with open(filename, "w", encoding=encoding) as f:
        f.write("[PARAMETERS]\n")
        for key, value in param_defaults.items():
            f.write(f"{key}={value}\n")
        f.write("[LABELS]\n")
        for label, name in zip(labels, names):
            f.write(f"{param_defaults['BYTE_PER_SAMPLE'] * param_defaults['N_CHANNEL'] * label},")
            f.write(f"{level_to_code['B1']},")
            f.write(f"{name}\n")