"""A lost preprocessing worker must fail the caller instead of hanging."""
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


def _exit_during_batch(*args, **kwargs):
    import fastdb4py as fdb

    # The child has started and attached to the actual preprocessing database.
    db = fdb.RecordEngine.load(kwargs['fdb_fn'])
    os._exit(19)


@pytest.mark.parametrize('builder', ['ne', 'ns'])
def test_preprocess_worker_exit_raises_within_deadline(tmp_path, builder):
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), builder, str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        # Bound a regression of this test without leaving its workers behind.
        os.killpg(process.pid, signal.SIGKILL)
        stdout, _ = process.communicate(timeout=5)
        pytest.fail(f'Preprocessing failed to report worker exit within 15 s:\n{stdout}')
    assert process.returncode == 0, stdout
    assert 'BROKEN_PROCESS_POOL' in stdout


def _run_failed_preprocess(builder, directory):
    from concurrent.futures.process import BrokenProcessPool
    from src.nh_flood_2d.preprocess import domain

    path = Path(directory)
    source = path / f'{builder}.txt'
    source.write_text('1,1,1,1,1,1,2,3,4,5,5,0,10\n' if builder == 'ne'
                      else '1,2,0,1,0,0,10,0,5,0,0\n')
    setattr(domain, f'_batch_{builder}_worker', _exit_during_batch)
    try:
        getattr(domain, f'create_{builder}_fdb_parallel')(str(source), str(path / f'{builder}.fdb'))
    except BrokenProcessPool:
        print('BROKEN_PROCESS_POOL', flush=True)
        return
    raise AssertionError('Worker exit did not raise BrokenProcessPool')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    _run_failed_preprocess(*sys.argv[1:])
