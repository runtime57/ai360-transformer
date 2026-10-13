import json
from collections import OrderedDict
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt", encoding="utf-8") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_txt(fname):
    fname = Path(fname)
    return fname.read_text(encoding="utf-8")


def write_txt(content, fname):
    fname = Path(fname)
    fname.write_text(content, encoding="utf-8")
