"""Shared raw-data verification and safe artifact extraction for all experiments."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

PROJECT_DIR = Path(__file__).resolve().parents[2]


def verify_training_data(data_dir):
    expected = json.loads((PROJECT_DIR / 'configs/data_source_hashes.json').read_text())['files']
    actual = {}
    for name, reference in expected.items():
        path = Path(data_dir) / name
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        actual[name] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
        if actual[name] != reference:
            raise ValueError(f'Training input differs from fixed local data: {name}.')
        print('VERIFIED TRAINING INPUT', name, actual[name]['sha256'], flush=True)
    (PROJECT_DIR / 'data_source_check.json').write_text(
        json.dumps({'status': 'verified', 'files': actual}, indent=2), encoding='utf-8')


def extract_archive(archive: Path, destination: Path):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Import into a fresh directory; preserve previous CPU/GPU runs.')
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            relative = PurePosixPath(entry.filename)
            target = destination.joinpath(*relative.parts).resolve()
            if (relative.is_absolute() or chr(92) in entry.filename or ':' in entry.filename
                    or '..' in relative.parts or destination.resolve() not in target.parents):
                raise ValueError('Archive path escapes its destination.')
        destination.mkdir(parents=True, exist_ok=True)
        bundle.extractall(destination)
