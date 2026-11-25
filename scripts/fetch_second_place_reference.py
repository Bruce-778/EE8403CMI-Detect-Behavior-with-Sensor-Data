"""Fetch the exact public author's source snapshot into an ignored reference folder."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = json.loads((ROOT / 'configs/second_place_source.json').read_text())
    directory = ROOT / 'outputs/reference_code/second_place'
    directory.mkdir(parents=True, exist_ok=True)
    base = 'https://raw.githubusercontent.com/Yamato-Arai/kaggle-cmi-detect-behavior-2nd-place-solution/' + source['commit']
    for name, expected in source['files'].items():
        path = directory / name
        if path.exists():
            actual = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            if actual != expected:
                raise ValueError(f'Existing source differs; preserve it: {path}')
            continue
        with urllib.request.urlopen(base + '/' + name, timeout=60) as response:
            content = response.read()
        actual = hashlib.sha256(content.replace(b'\r\n', b'\n')).hexdigest()
        if actual != expected:
            raise ValueError(f'Upstream download does not match pinned source: {name}')
        path.write_bytes(content)
    print('Verified pinned source', source['commit'], directory)


if __name__ == '__main__':
    main()
