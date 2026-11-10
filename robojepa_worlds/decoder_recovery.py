"""Numerical recovery diagnostics; only imported inside GPU workers."""
import torch


def state_difference(expected, actual):
    """Compare nested model/optimizer states without hashing tensor contents."""
    result = {'exact': True, 'max_error': 0., 'worst_key': None, 'mismatched_leaves': 0}

    def compare(left, right, path):
        if isinstance(left, torch.Tensor):
            right = right.detach().cpu()
            left = left.detach().cpu()
            equal = left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
            error = float((left.double()-right.double()).abs().max()) if left.shape == right.shape and left.numel() else 0.
            if not equal and (result['worst_key'] is None or error > result['max_error']):
                result.update(max_error=error, worst_key=path)
        elif isinstance(left, dict):
            if left.keys() != right.keys():
                raise ValueError('Recovery state keys changed: ' + path)
            for key in left:
                compare(left[key], right[key], path + '/' + str(key))
            return
        elif isinstance(left, (tuple, list)):
            if len(left) != len(right):
                raise ValueError('Recovery state length changed: ' + path)
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, path + '/' + str(index))
            return
        else:
            equal = left == right
        if not equal:
            result['exact'] = False
            result['mismatched_leaves'] += 1

    compare(expected, actual, '')
    return result
