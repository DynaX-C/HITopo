import json
import argparse
from types import SimpleNamespace


def load_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    return SimpleNamespace(**cfg)


def _convert_value(value, ref_value):
    if isinstance(ref_value, bool):
        if isinstance(value, bool):
            return value
        value = str(value).lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f"Cannot convert {value} to bool")
    if isinstance(ref_value, int) and not isinstance(ref_value, bool):
        return int(value)
    if isinstance(ref_value, float):
        return float(value)
    if isinstance(ref_value, list):
        return json.loads(value)
    if isinstance(ref_value, dict):
        return json.loads(value)
    return value


def override_args(args, unknown_args):
    """
    unknown_args example:
        ["--lr", "1e-3", "--batch_size", "64", "--variant", "full"]
    """
    if len(unknown_args) % 2 != 0:
        raise ValueError(f"Unrecognized CLI arguments must be in '--key value' pairs: {unknown_args}")

    i = 0
    while i < len(unknown_args):
        key = unknown_args[i]
        value = unknown_args[i + 1]

        if not key.startswith("--"):
            raise ValueError(f"Invalid argument format: {key}")

        key = key[2:]

        if not hasattr(args, key):
            raise ValueError(f"Unknown config key for override: {key}")

        ref_value = getattr(args, key)
        new_value = _convert_value(value, ref_value)
        setattr(args, key, new_value)

        i += 2

    return args


def parse_config_and_override():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    known_args, unknown_args = parser.parse_known_args()

    args = load_config(known_args.config)
    args = override_args(args, unknown_args)

    return args