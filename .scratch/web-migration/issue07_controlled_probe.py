import gc,hashlib,io,json,time,zipfile
from pathlib import Path
from uuid import uuid4
import numpy as np, torch,soundfile as sf
from fastapi.testclient import TestClient
from backend.app import create_app
from backend.config import Settings
from backend.auth import hash_password
from backend.generation_compute import scientific_generation
from backend.worker import run_worker
from auditory_cortex.optimal_stimulus.generator import Sampler
from auditory_cortex.diffusion.utils import seed_everything
import auditory_cortex.io_utils.io as aio
old=json.loads(Path('.scratch/web-migration/issue05-parity.json').read_text())
root=Path(old['root']);fit_id=old['attempts'][-1]['fit_id']
settings=Settings(root/'sessions.sqlite3','operator',hash_password('test-password'))
client=TestClient(create_app(settings));client.auth=('operator','test-password')
output=Path('/scratch/gilbreth/ahmedb/cache/auditory-web-checks/issue07-controlled')/str(uuid4());output.mkdir(parents=True)
aio.cache_dir=Path('/scratch/gilbreth/ahmedb/cache/web-migration-baseline/issue-01/fit-cache')
channels=json.loads((root/'fits'/fit_id/'trf_config.json').read_text())['unit_ids']
report={'completed':False,'historical_exact_equality_passed':False,'acceptance':'User-approved controlled legacy/web initialization and RNG comparison','root':str(output),'results':[]}
for objective in ['stretch','one-hot']:
    seed_everything(42)
    rng={'cpu_sha256':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),'cuda_sha256':hashlib.sha256(torch.cuda.get_rng_state(0).cpu().numpy().tobytes()).hexdigest()}
    sampler=Sampler(torch.device('cuda:0'),model_name='deepspeech2',session_id=0,layer_id=2,bin_width=50,mVocs=False,lag=200,dataset_name='ucdavisBAK',unit_ids=channels)
    files=sampler.generate_and_save_stimuli(output/objective,unit_id=12.03,task=objective,duration=1,n_stimuli=1,seed=42,stim_type='timit')
    legacy=Path(files[0]['path']);del sampler;gc.collect();torch.cuda.empty_cache()
    response=client.post('/api/v1/fits/'+fit_id+'/generations',json={'request_id':str(uuid4()),'channel_id':'12.03','objective':objective,'duration_seconds':1,'clip_count':1,'seed':42})
    assert response.status_code==202,response.text
    job=response.json()
    def compute(task,progress):
        progress('generating');scientific_generation(task.workspace);gc.collect();torch.cuda.empty_cache();return 0
    run_worker(settings,once=True,generation_runner=compute)
    final=client.get('/api/v1/generations/'+job['id']).json();assert final['state']=='succeeded',final
    provenance=json.loads((root/'generations'/job['id']/'compute-provenance.json').read_text())
    assert provenance['initialization_rng']==rng,(rng,provenance['initialization_rng'])
    response=client.get('/api/v1/generations/'+job['id']+'/download');assert response.status_code==200,response.text
    archive=zipfile.ZipFile(io.BytesIO(response.content));data=archive.read('clip-001.wav')
    expected=legacy.read_bytes();a,_=sf.read(io.BytesIO(data));b,_=sf.read(legacy)
    entry={'objective':objective,'generation_id':job['id'],'job_id':job['job_id'],'fit_id':fit_id,'legacy_path':str(legacy),'web_sha256':hashlib.sha256(data).hexdigest(),'legacy_sha256':hashlib.sha256(expected).hexdigest(),'initialization_rng':rng,'max_absolute_pcm_difference':float(np.max(abs(a-b))),'rms_pcm_difference':float(np.sqrt(np.mean((a-b)**2))),'frames':len(a),'exit_code':final['exit_code'],'byte_identical':data==expected}
    report['results'].append(entry);Path('.scratch/web-migration/issue07-parity.json').write_text(json.dumps(report,indent=2)+'\n');print('CONTROLLED_COMPARISON',json.dumps(entry),flush=True)
    assert data==expected,entry
report['completed']=True
Path('.scratch/web-migration/issue07-parity.json').write_text(json.dumps(report,indent=2)+'\n')
