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
        import cupy as cp
        with cp.cuda.Device(device):
            if cp.arange(10).sum().item() != 45:
                raise RuntimeError('CuPy CUDA kernel check failed.')
            answer = cp.linalg.solve(cp.asarray([[3., 1.], [1., 2.]]), cp.asarray([9., 8.]))
            if not bool(cp.allclose(answer, cp.asarray([2., 3.])).item()):
                raise RuntimeError('CuPy CUDA linear algebra check failed.')
            cp.cuda.get_current_stream().synchronize()
        print('PyTorch and CuPy CUDA ready: GPU %d (%s)' % (device, torch.cuda.get_device_name(device)), flush=True)


if __name__ == '__main__':
    try:
        check_gpu_access()
    except Exception as error:
        raise SystemExit('GPU preflight failed. Use a working Slurm GPU allocation and the configured CUDA runtime. '
                         'See docs/deployment/python-runtime.md.\n' + str(error))
