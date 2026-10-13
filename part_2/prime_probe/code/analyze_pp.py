"""Validate a run_pp_lab campaign and produce Task 2.2 tables and figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODES = ('signal', 'no_victim', 'unrelated')
DTYPE = [('phase', 'U11'), ('mode', 'U10'), ('run', 'i4'), ('trial', 'i4'),
         ('bit', 'i1'), ('probe_order', 'i1'), ('time_a_ticks', 'i8'),
         ('time_b_ticks', 'i8'), ('latency_ns', 'i8'), ('guess', 'i1')]
BASE_ARTIFACTS = {'pp_measure.c', 'metadata.json', 'cpuinfo.txt'}
MEASUREMENT_ARTIFACTS = {'pp_measure', 'build.log', 'assembly.txt', 'raw.csv', 'events.jsonl'}
DEMO_ARTIFACTS = {'pp_demo.c', 'pp_demo', 'demo_build.log', 'demo_assembly.txt',
                  'demo_bit0.csv', 'demo_bit0.log', 'demo_bit1.csv', 'demo_bit1.log'}
LOCAL_RECORDS = {'hashes.json', 'collection_status.json', 'ssh.log'}


def validate_host_artifacts(folder, record):
    """Check collection provenance and required hashes before reading measurements."""
    if record.get('status') not in ('collected', 'unsupported'):
        raise ValueError('Collection did not complete: ' + str(record.get('status')))
    if record.get('transfer_verified') is not True:
        raise ValueError('Artifact transfer was not verified')
    if record.get('host') != folder.name:
        raise ValueError('Campaign host does not match artifact folder')
    collection_file = folder / 'collection_status.json'
    if collection_file.exists():
        collection = json.loads(collection_file.read_text())
        if not isinstance(collection, dict):
            raise ValueError('Collection record must be an object')
        for key in ('host', 'status', 'transfer_verified'):
            if collection.get(key) != record.get(key):
                raise ValueError('Collection record disagrees with campaign: ' + key)
    hashes = json.loads((folder / 'hashes.json').read_text())
    if not isinstance(hashes, dict):
        raise ValueError('Artifact manifest must be a hash mapping')
    meta = json.loads((folder / 'metadata.json').read_text())
    if not isinstance(meta, dict):
        raise ValueError('Metadata must be an object')
    if meta.get('host') != folder.name or meta.get('status') != record.get('status'):
        raise ValueError('Metadata host/status disagrees with campaign')
    required = BASE_ARTIFACTS.copy()
    has_demo = any(key in meta for key in ('demo_source_sha256', 'demo_executable_sha256', 'demo_runs'))
    if has_demo:
        required.add('pp_demo.c')
    if meta['status'] == 'collected':
        required.update(MEASUREMENT_ARTIFACTS)
        if has_demo:
            required.update(DEMO_ARTIFACTS)
            runs = meta.get('demo_runs')
            if (not isinstance(runs, list) or len(runs) != 2 or
                    not all(isinstance(run, dict) and run.get('returncode') == 0 for run in runs) or
                    sorted(run.get('bit', -1) for run in runs) != [0, 1]):
                raise ValueError('Demo metadata must record two successful bit runs')
    missing = required - hashes.keys()
    if missing:
        raise ValueError('Missing required manifest entries: ' + ', '.join(sorted(missing)))
    uncovered = {p.name for p in folder.iterdir() if p.is_file()} - LOCAL_RECORDS - hashes.keys()
    if uncovered:
        raise ValueError('Artifacts missing hash coverage: ' + ', '.join(sorted(uncovered)))
    for name, digest in hashes.items():
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
            raise ValueError('Invalid artifact name')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Invalid artifact hash: ' + name)
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Artifact changed: {folder.name}/{name}')
    if meta.get('source_sha256') != hashes['pp_measure.c']:
        raise ValueError('Source hash disagrees with metadata')
    if meta['status'] == 'collected' and meta.get('executable_sha256') != hashes['pp_measure']:
        raise ValueError('Executable hash disagrees with metadata')
    if has_demo and meta.get('demo_source_sha256') != hashes['pp_demo.c']:
        raise ValueError('Demo source hash disagrees with metadata')
    if has_demo and meta['status'] == 'collected' and meta.get('demo_executable_sha256') != hashes['pp_demo']:
        raise ValueError('Demo executable hash disagrees with metadata')
    return meta


def analyze_record(campaign, record, figures=None):
    """Analyze one campaign record, retaining collection/validation failures as statuses."""
    host = record.get('host', '')
    status = record.get('status', 'unavailable_or_failed')
    if status not in ('collected', 'unsupported'):
        return {'host': host, 'status': status, 'reason': record.get('reason')}, []
    if record.get('transfer_verified') is not True:
        return {'host': host, 'status': 'unverified',
                'reason': record.get('reason') or 'Artifact transfer was not verified'}, []
    if not isinstance(host, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', host):
        return {'host': host, 'status': 'validation_failed', 'reason': 'Invalid campaign host'}, []
    try:
        return analyze_host(campaign / host, figures, record=record)
    except (OSError, ValueError, AssertionError, KeyError, TypeError, StopIteration) as error:
        reason = str(error) or type(error).__name__
        return {'host': host, 'status': 'validation_failed', 'reason': reason}, []


def metrics(bits, guesses):
    zero, one = bits == 0, bits == 1
    if not np.any(zero) or not np.any(one):
        raise ValueError('Both ground-truth classes are required')
    return {'accuracy': float(np.mean(bits == guesses)),
            'false_positive_rate': float(np.mean(guesses[zero] == 1)),
            'false_negative_rate': float(np.mean(guesses[one] == 0))}


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save_plot(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / (name + '.png'), dpi=170)
    fig.savefig(directory / (name + '.pdf'))
    plt.close(fig)


def analyze_host(folder, figures=None, record=None):
    if record is None:
        record = json.loads((folder / 'collection_status.json').read_text())
    meta = validate_host_artifacts(folder, record)
    if meta['status'] != 'collected':
        return {'host': folder.name, 'status': meta['status'], 'reason': meta.get('reason'),
                'metadata': meta}, []
    data = np.genfromtxt(folder / 'raw.csv', delimiter=',', names=True,
                         dtype=DTYPE, encoding='utf-8')
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
    calibration = [e for e in events if e['type'] == 'calibration']
    setup = [e for e in events if e['type'] == 'setup']
    assert len(calibration) == len(setup) == 1
    calibration, setup = calibration[0], setup[0]
    assert calibration['samples_per_bit'] == 10000
    assert len(data) == 320000, 'Expected 20,000 calibration + 300,000 evaluation rows'
    assert np.all(np.isin(data['bit'], [0, 1]))
    assert np.all(np.isin(data['probe_order'], [0, 1]))
    assert np.all(data['time_a_ticks'] > 0) and np.all(data['time_b_ticks'] > 0)
    assert np.all(data['latency_ns'] > 0)
    cal = data[data['phase'] == 'calibration']
    assert len(cal) == 20000 and np.all(cal['guess'] == -1)
    assert np.all(cal['mode'] == 'signal') and np.all(cal['run'] == -1)
    assert np.array_equal(cal['trial'], np.arange(20000))
    score = cal['time_b_ticks'] - cal['time_a_ticks']
    medians = []
    for bit in (0, 1):
        assert np.count_nonzero(cal['bit'] == bit) == 10000
        medians.append(float(np.median(score[cal['bit'] == bit])))
    assert medians == [calibration['median0'], calibration['median1']]
    assert (medians[0] + medians[1]) / 2 == calibration['threshold']
    evaluation = data[data['phase'] == 'evaluation']
    assert len(evaluation) == 300000
    predictions = (evaluation['time_b_ticks'] - evaluation['time_a_ticks']) > calibration['threshold']
    assert np.array_equal(predictions.astype(int), evaluation['guess']), 'Threshold changed or guesses disagree'
    batches = {(e['mode'], e['run']): e for e in events if e['type'] == 'batch'}
    assert len(batches) == 30
    rows = []
    summary = {'host': folder.name, 'status': 'validated', 'metadata': meta,
               'setup': setup, 'calibration': calibration,
               'resources': next(e for e in events if e['type'] == 'resources'), 'modes': {}}
    for mode in MODES:
        samples = evaluation[evaluation['mode'] == mode]
        elapsed = 0
        accuracies = []
        for run in range(10):
            group = samples[samples['run'] == run]
            assert len(group) == 10000
            assert np.array_equal(group['trial'], np.arange(10000))
            assert np.count_nonzero(group['bit'] == 0) == 5000
            assert np.count_nonzero(group['bit'] == 1) == 5000
            event = batches[mode, run]
            assert event['trials'] == 10000 and event['cpu_after'] == setup['cpu']
            assert event['elapsed_ns'] >= np.sum(group['latency_ns'])
            result = metrics(group['bit'], group['guess'])
            rate = 10000 * 1e9 / event['elapsed_ns']
            row = {'host': folder.name, 'mode': mode, 'run': run, **result,
                   'trials_per_second': rate,
                   'correct_guesses_per_second': result['accuracy'] * rate,
                   'median_latency_ns': float(np.median(group['latency_ns'])),
                   'p95_latency_ns': float(np.percentile(group['latency_ns'], 95))}
            rows.append(row)
            elapsed += event['elapsed_ns']
            accuracies.append(result['accuracy'])
        result = metrics(samples['bit'], samples['guess'])
        rate = len(samples) * 1e9 / elapsed
        summary['modes'][mode] = {**result, 'samples': len(samples),
            'trials_per_second': rate, 'correct_guesses_per_second': result['accuracy'] * rate,
            'median_latency_ns': float(np.median(samples['latency_ns'])),
            'p95_latency_ns': float(np.percentile(samples['latency_ns'], 95)),
            'run_accuracy_min': min(accuracies), 'run_accuracy_max': max(accuracies),
            'run_accuracy_sd': float(np.std(accuracies, ddof=1)),
            'a_targeted_minus_other_median_ticks': float(np.median(samples['time_a_ticks'][samples['bit'] == 0]) - np.median(samples['time_a_ticks'][samples['bit'] == 1])),
            'b_targeted_minus_other_median_ticks': float(np.median(samples['time_b_ticks'][samples['bit'] == 1]) - np.median(samples['time_b_ticks'][samples['bit'] == 0]))}

    if figures is not None:
        plot_host(folder, figures, evaluation, calibration)
    return summary, rows


def plot_host(folder, figures, evaluation, calibration):

    # Each host keeps its own TSC units. Clipping affects only the plotted window.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, mode in zip(axes, MODES):
        group = evaluation[evaluation['mode'] == mode]
        difference = group['time_b_ticks'] - group['time_a_ticks']
        lo, hi = np.percentile(difference, [0.5, 99.5])
        if lo == hi: lo, hi = lo - 1, hi + 1
        bins = np.linspace(lo, hi, 65)
        for bit, color in [(0, '#2563eb'), (1, '#dc2626')]:
            ax.hist(difference[group['bit'] == bit], bins=bins, density=True,
                    histtype='step', color=color, label=f'bit {bit}')
        ax.axvline(calibration['threshold'], color='black', ls='--', label='frozen threshold')
        ax.set(title=mode, xlabel='Probe B - probe A (TSC ticks)', ylabel='Density in shown window')
        ax.legend(fontsize=7)
    fig.suptitle(f'{folder.name}: timing distributions (central 99%; all samples retained)')
    save_plot(fig, figures, folder.name + '_distributions')

    group = evaluation[(evaluation['mode'] == 'signal') & (evaluation['run'] == 0)][:200]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(group['trial'], group['time_a_ticks'], label='Probe A', lw=0.8)
    axes[0].plot(group['trial'], group['time_b_ticks'], label='Probe B', lw=0.8)
    axes[0].set_ylabel('TSC ticks'); axes[0].legend()
    axes[1].plot(group['trial'], group['time_b_ticks'] - group['time_a_ticks'], lw=0.8)
    axes[1].axhline(calibration['threshold'], color='black', ls='--')
    axes[1].set_ylabel('B - A ticks')
    axes[2].step(group['trial'], group['bit'], where='mid', label='Known bit')
    axes[2].scatter(group['trial'], group['guess'], s=8, color='#dc2626', label='Guess')
    axes[2].set(xlabel='Trial (first 200 of signal run 0)', ylabel='Bit', yticks=[0, 1]); axes[2].legend()
    fig.suptitle(folder.name + ': Prime+Probe trace')
    save_plot(fig, figures, folder.name + '_trace')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    output = args.campaign / 'analysis'
    output.mkdir(exist_ok=True)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    summaries, run_rows, table = [], [], []
    for record in json.loads((args.campaign / 'campaign.json').read_text()):
        summary, rows = analyze_record(args.campaign, record, figures)
        summaries.append(summary); run_rows.extend(rows)
        print(summary['host'], summary['status'],
              {k: round(v['accuracy'], 4) for k, v in summary.get('modes', {}).items()}, flush=True)
        for mode, m in summary.get('modes', {}).items():
            table.append({'host': summary['host'], 'mode': mode, **m,
                          'setup_ns': summary['setup']['setup_ns'],
                          'calibration_ns': summary['calibration']['elapsed_ns'],
                          'attack_allocation_bytes': summary['setup']['attack_allocation_bytes'],
                          'max_rss_kib': summary['resources']['max_rss_kib']})
    write_csv(output / 'per_run.csv', run_rows)
    write_csv(output / 'summary.csv', table)
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    fig, ax = plt.subplots(figsize=(10, 4.5))
    supported = [s for s in summaries if s['status'] == 'validated']
    for index, summary in enumerate(supported):
        for mode, color, offset in zip(MODES, ['#2563eb', '#777777', '#d97706'], [-0.2, 0, 0.2]):
            values = [100 * r['accuracy'] for r in run_rows if r['host'] == summary['host'] and r['mode'] == mode]
            ax.scatter(np.full(len(values), index + offset), values, s=17, color=color,
                       alpha=0.7, label=mode if index == 0 else None)
    ax.axhline(50, color='black', ls='--', lw=1)
    ax.set(xticks=range(len(supported)), xticklabels=[s['host'] for s in supported],
           ylabel='Accuracy (%)', title='Ten repeated batches per condition; balanced labels')
    if supported:
        ax.legend()
    save_plot(fig, figures, 'run_variation')


if __name__ == '__main__':
    main()
