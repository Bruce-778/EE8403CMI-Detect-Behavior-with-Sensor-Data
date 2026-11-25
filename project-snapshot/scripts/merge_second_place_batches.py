"""Merge verified single-fold archives without sharing joint axes across folds.

Re-extract the byte-verified original downloads, preserve every arm and routed
fold directory unchanged, and run the full auditor again on the merged archive.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from evaluate_second_place import ROOT, dump
from import_winner_experiments import extract_archive


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_records(records):
    if len(records) != 5 or sorted(r.get('folds', []) for r in records) != [[i] for i in range(5)]:
        raise ValueError('Exactly one verified batch for each fixed fold is required.')
    for record in records:
        if record.get('status') != 'verified' or record.get('architectures') != ['base']:
            raise ValueError('Only individually verified base batches can be merged.')
        fold = record['folds'][0]
        if sorted((a['architecture'], a['variant'], a['fold']) for a in record['arms']) != [
                ('base', v, fold) for v in ('all', 'all_rot', 'imu', 'imu_rot')]:
            raise ValueError('Each batch must contain its four unique branches.')
        for key in ('source_commit', 'folds_sha256', 'training_inputs', 'metadata_sha256', 'orders'):
            if record[key] != records[0][key]:
                raise ValueError(f'Batch {key} differs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, nargs=5, required=True)
    parser.add_argument('--archives', type=Path, nargs=5, required=True)
    parser.add_argument('--v1-metadata', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    records = [json.loads(p.read_text(encoding='utf-8')) for p in args.records]
    validate_records(records)
    work = (ROOT / args.work_dir).resolve()
    if (ROOT / 'outputs/second_place').resolve() not in work.parents or work.exists():
        raise ValueError('Use a new directory below outputs/second_place.')
    if sha(args.v1_metadata) != records[0]['metadata_sha256']:
        raise ValueError('Original raw-order metadata bytes differ.')
    for archive, record in zip(args.archives, records):
        if sha(archive) != record['archive_sha256']:
            raise ValueError('Original source archive bytes differ from verified record.')
    work.mkdir(parents=True)
    package = work / 'package'
    package.mkdir()
    run = package / 'outputs/second_place/base_five_fold_merged'
    run.mkdir(parents=True)
    batches, copied = [], []
    for i, (archive, record, record_path) in enumerate(zip(args.archives, records, args.records)):
        fold = record['folds'][0]
        extracted = work / f'original_batch_fold_{fold}'
        extract_archive(archive.resolve(), extracted)
        arms = sorted(extracted.glob('outputs/second_place/*/base/*/fold_*/provenance.json'))
        if len(arms) != 4 or any(p.parents[3] != arms[0].parents[3] for p in arms):
            raise ValueError('Original batch experiment root is ambiguous.')
        original_run = arms[0].parents[3]
        metadata = original_run / 'cache/metadata.csv'
        if not metadata.exists():
            if fold != 0:
                raise ValueError('Only v1 fold0 may need externally recovered metadata.')
            metadata = args.v1_metadata
        if sha(metadata) != record['metadata_sha256']:
            raise ValueError('Source raw-order metadata differs.')
        if i == 0:
            shutil.copytree(extracted / 'configs', package / 'configs')
            shutil.copy2(extracted / 'data_source_check.json', package / 'data_source_check.json')
            (run / 'cache').mkdir()
            shutil.copy2(metadata, run / 'cache/metadata.csv')
        else:
            for name in ('folds.csv', 'second_place_source.json'):
                if sha(extracted / 'configs' / name) != sha(package / 'configs' / name):
                    raise ValueError('Frozen source or fold file bytes differ between batches.')
            if json.loads((extracted / 'data_source_check.json').read_text())['files'] != record['training_inputs']:
                raise ValueError('Original batch raw input evidence differs.')
        directories = []
        for path in arms:
            provenance = json.loads(path.read_text())
            if provenance['fold'] != fold or provenance['seed'] != 42 + fold:
                raise ValueError('Actual fold or original seed differs.')
            directories.append(path.parent)
        directories.extend(original_run / 'routed' / scenario / f'fold_{fold}'
                           for scenario in ('observed', 'aux_dropout50', 'imu_only'))
        for directory in directories:
            relative = directory.relative_to(original_run)
            target = run / relative
            if target.exists():
                raise ValueError('Cross-batch destination collision.')
            shutil.copytree(directory, target)
            for path in sorted(directory.rglob('*')):
                if path.is_file():
                    new_path = target / path.relative_to(directory)
                    digest = sha(path)
                    if sha(new_path) != digest:
                        raise ValueError('Copied evidence changed.')
                    copied.append({'fold': fold, 'path': new_path.relative_to(package).as_posix(), 'sha256': digest})
        batches.append({'fold': fold, 'source_url': record['source_url'], 'archive': str(archive.resolve()),
                        'archive_sha256': record['archive_sha256'], 'archive_bytes': archive.stat().st_size,
                        'record': str(record_path.resolve()), 'record_sha256': sha(record_path),
                        'metadata_sha256': record['metadata_sha256'], 'seed': 42 + fold})
    source_manifest = {'source_batches': batches, 'copied_files': copied,
                       'protocol': 'Unchanged per-fold joint ontology, train IDs, seed and last weights; no retraining.'}
    dump(package / 'merge_provenance.json', source_manifest)
    merged_zip = work / 'second_place_merged_five_fold.zip'
    with zipfile.ZipFile(merged_zip, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(package.rglob('*')):
            if path.is_file():
                bundle.write(path, path.relative_to(package).as_posix())
    subprocess.run([sys.executable, '-s', str(ROOT / 'scripts/evaluate_second_place.py'),
                    '--archive', str(merged_zip), '--output-dir', str(args.output_dir),
                    '--name', args.name, '--source-url', 'Local verified merge; source URLs in merge_provenance.json'],
                   check=True, cwd=ROOT)
    record_path = ROOT / f'experiments/results/{args.name}.json'
    result = json.loads(record_path.read_text())
    # Verify re-audited decisions match each individually completed experiment.
    for scenario, combined in result['scenarios'].items():
        for record in records:
            fold = record['folds'][0]
            row = next(r for r in combined['folds'] if r['fold'] == fold)
            if row != record['scenarios'][scenario]['folds'][0]:
                raise ValueError('Merged re-audit differs from original scored evidence.')
    result['source_batches'] = batches
    result['merge_provenance_sha256'] = sha(package / 'merge_provenance.json')
    dump(record_path, result)
    dump(ROOT / args.output_dir / 'import_check.json', result)
    print('Verified five-fold merge and original batch scores:', record_path, flush=True)


if __name__ == '__main__':
    main()
