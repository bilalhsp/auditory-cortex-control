import threading,time
from dataclasses import replace
from pathlib import Path
from backend.app import create_app
from backend.tests.test_recordings import RecordingApiTests
from backend.tests.test_fits import FitApiTests
from backend.tests.test_generations import GenerationApiTests
from backend.worker import run_worker
root=Path('/scratch/gilbreth/ahmedb/cache/auditory-web-checks')
trigger=root/'issue11-trigger';failure=root/'issue11-fail';batchmode=root/'issue11-batch'
for path in (trigger,failure,batchmode):path.unlink(missing_ok=True)
fixture=RecordingApiTests();fixture.setUpClass();fixture.setUp()
settings=replace(fixture.settings,allowed_origins=('http://127.0.0.1:5173',))
app=create_app(settings,lambda:1)
def pause(task,seconds):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if task.cancelled():return False
        time.sleep(.1)
    return True
def fit_runner(task,progress):
    progress('fitting')
    if not pause(task,8):raise RuntimeError('Stopped')
    return FitApiTests.output(task,progress)
def runner(task,progress):
    progress('generating')
    delay=(30 if task.generation.objective=='one-hot' else 4) if batchmode.exists() else 12
    if not pause(task,delay):return -15
    if failure.exists() and failure.read_text()=='yes':return 7
    return GenerationApiTests.compute(task,progress)
def worker():
    while not trigger.exists():time.sleep(.1)
    run_worker(settings,runner=fit_runner,generation_runner=runner,gpu_count=lambda:1)
threading.Thread(target=worker,daemon=True).start()
