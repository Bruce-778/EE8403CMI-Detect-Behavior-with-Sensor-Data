"""Build and attest the local raw-input cache for causal adaptation evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cmi_project.second_place import build_cache, load_reference
from cmi_project.validation import load_fold_manifest
from run_winner_experiments import verify_training_data
from merge_second_place_batches import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cache = (ROOT / args.output).resolve()
    if (ROOT / 'outputs/second_place').resolve() not in cache.parents or cache.exists():
        raise ValueError('Use a fresh cache under outputs/second_place.')
    start = time.perf_counter()
    verify_training_data(ROOT / 'data')
    reference = load_reference(ROOT / 'outputs/reference_code/second_place', ROOT / 'configs/second_place_source.json')
    build_cache(ROOT / 'data', cache, load_fold_manifest(ROOT / 'configs/folds.csv'), reference)
    metadata = pd.read_csv(cache / 'metadata.csv')
    original = pd.read_csv(ROOT / 'outputs/second_place/recovery_v1/metadata.csv')
    # Column formatting may differ across pandas versions; compare the actual
    # order, fields and values, never infer row order from a sorted CSV.
    pd.testing.assert_frame_equal(metadata, original, check_dtype=False)
    attestation = {'status': 'verified', 'seconds': time.perf_counter() - start,
        'sequences': len(metadata), 'source_commit': reference.provenance['commit'],
        'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(cache.iterdir()) if p.is_file()},
        'original_cloud_metadata_sha256': sha(ROOT / 'outputs/second_place/recovery_v1/metadata.csv'),
        'training_inputs': json.loads((ROOT / 'data_source_check.json').read_text())['files'],
        'protocol': 'Full source physical features from byte-verified raw training CSV; raw sequence order matches cloud metadata. Validation annotations are excluded from the online inference interface.'}
    (cache / 'cache_check.json').write_text(json.dumps(attestation, indent=2), encoding='utf-8')
    print(json.dumps(attestation, indent=2), flush=True)


if __name__ == '__main__':
    main()
