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


def check_gpu_access() -> None:
    """Verify actual CUDA operations, including MPS access, on visible devices."""
    import torch
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError('No CUDA GPUs are visible.')
    for device in range(count):
        torch.ones(1, device='cuda:%d' % device)
        torch.cuda.synchronize(device)
        print('CUDA ready: GPU %d (%s)' % (device, torch.cuda.get_device_name(device)), flush=True)


if __name__ == '__main__':
    try:
        check_gpu_access()
    except (ImportError, RuntimeError, OSError) as error:
        raise SystemExit('GPU preflight failed. Use a working Slurm GPU allocation and check CUDA/MPS access.\n' + str(error))
