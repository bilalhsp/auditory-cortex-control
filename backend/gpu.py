"""Backend-local GPU visibility; no remote hosts or browser device identifiers."""
import os


def local_gpu_count() -> int:
    import torch
    return torch.cuda.device_count()


def visibility_signature():
    return os.environ.get('CUDA_VISIBLE_DEVICES')


def device_environment(index: int) -> dict:
    if index < 0:
        raise ValueError('Invalid local GPU index')
    env = dict(os.environ)
    visible = visibility_signature()
    if visible is not None:
        devices = visible.split(',')
        if index >= len(devices) or not devices[index].strip() or devices[index].strip() == '-1':
            raise ValueError('GPU is outside runtime visibility')
        env['CUDA_VISIBLE_DEVICES'] = devices[index].strip()
    else:
        env['CUDA_VISIBLE_DEVICES'] = str(index)
    return env
