"""Evaluation controls in the original Capsaicin incoming-frame convention."""
import statistics

from .capsaicin import ACTION_CHANNELS, chunk_rows

CONTROL_NAMES = ('recorded_then_stop', 'zero', 'forward', 'backward', 'yaw_negative', 'yaw_positive')


def training_yaw(index):
    values = [abs(row[8]) for clip in index['clips'] if clip['split'] == 'train'
              for row in clip['frame_actions'][1:] if abs(row[8]) > 1e-8]
    if not values:
        raise ValueError('No nonzero training yaw')
    return {'degrees_per_source_frame': statistics.median(values), 'nonzero_frames': len(values),
            'selection': 'median absolute nonzero yaw, training split only'}


def controls(recorded, steps, yaw):
    zero = [0.] * 64
    result = {'recorded_then_stop': [list(x) for x in recorded[:steps]]}
    result['recorded_then_stop'] += [zero.copy() for _ in range(steps-len(result['recorded_then_stop']))]
    for name in CONTROL_NAMES[1:]:
        row = [0.] * len(ACTION_CHANNELS)
        if name in ('forward', 'backward'):
            row[ACTION_CHANNELS.index(name)] = 1.
        elif name.startswith('yaw_'):
            row[8] = yaw * (-1 if name == 'yaw_negative' else 1)
        _, result[name] = chunk_rows([[0.]*9] + [row.copy() for _ in range(steps*4)], steps*4+1)
    if any(len(rows) != steps or any(len(r) != 64 for r in rows) for rows in result.values()):
        raise ValueError('Invalid control horizon or channel count')
    return result
