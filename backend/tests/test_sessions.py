import unittest
from fastapi.testclient import TestClient
from backend.app import create_app
from backend.tests import test_fits


class SessionApiTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_fits.FitApiTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.settings, self.client = self.fixture.settings, self.fixture.client

    def test_resumable_sessions_prioritize_active_work_and_preserve_creation_timezone(self):
        legacy = self.fixture.experiment
        active = self.client.post(self.fixture.prefix + '/fits', json=self.fixture.request()).json()
        path = '/api/v1/sessions/resumable'
        response = self.client.get(path)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()[0]['session']['timezone'], 'America/Los_Angeles')

        experiment = self.client.post('/api/v1/experiments?session_timezone=Asia%2FTokyo',
                                      json={'gpu_count': 1}).json()
        session_id = experiment['session_id']
        replacement = self.client.post('/api/v1/experiments',
                                       params={'session_id': legacy['session_id'], 'session_timezone': 'UTC'},
                                       json={'gpu_count': 1}).json()
        restarted = TestClient(create_app(self.settings, lambda: 1))
        self.assertEqual(restarted.get(path).status_code, 401)
        restarted.auth = ('operator', 'test-password')
        rows = {row['session']['id']: row for row in restarted.get(path).json()}
        self.assertEqual(rows[session_id]['session']['timezone'], 'Asia/Tokyo')
        self.assertFalse(rows[session_id]['active_work'])
        self.assertEqual(rows[legacy['session_id']]['experiment']['id'], legacy['id'])
        self.assertTrue(rows[legacy['session_id']]['active_work'])
        self.assertEqual(rows[legacy['session_id']]['session']['timezone'], 'America/Los_Angeles')
        detail = restarted.get('/api/v1/sessions/' + legacy['session_id']).json()
        self.assertEqual({item['id'] for item in detail['experiments']}, {legacy['id'], replacement['id']})
        self.assertEqual(detail['jobs'][0]['id'], active['id'])
        restarted.post('/api/v1/sessions/' + legacy['session_id'] + '/close')
        self.assertEqual([row['session']['id'] for row in restarted.get(path).json()], [session_id])

    def test_reconnect_recovers_work_and_close_cancels_queued_job_idempotently(self):
        fit = self.client.post(self.fixture.prefix + '/fits', json=self.fixture.request()).json()
        response = self.client.get('/api/v1/sessions')
        self.assertEqual(response.status_code, 200)
        session = response.json()[0]
        self.assertNotEqual(session['id'], self.fixture.experiment['id'])
        restarted = TestClient(create_app(self.settings, lambda: 1))
        restarted.auth = ('operator', 'test-password')
        work = restarted.get(self.fixture.prefix + '/work').json()
        self.assertEqual(work['fits'][0]['id'], fit['id'])
        closed = restarted.post('/api/v1/sessions/' + session['id'] + '/close').json()
        self.assertEqual(closed['status'], 'closed')
        self.assertEqual(restarted.get('/api/v1/fits/' + fit['id']).json()['state'], 'cancelled')
        self.assertEqual(restarted.post('/api/v1/sessions/' + session['id'] + '/close').json(), closed)
        self.assertEqual(restarted.post(self.fixture.prefix + '/fits', json=self.fixture.request()).status_code, 409)

    def test_running_fit_keeps_reservation_until_compute_exits_and_stop_wins(self):
        import threading
        from backend.worker import run_worker
        fit = self.client.post(self.fixture.prefix + '/fits', json=self.fixture.request()).json()
        started, finish = threading.Event(), threading.Event()
        def compute(task, progress):
            started.set()
            if not finish.wait(10): raise RuntimeError('Controlled compute timed out')
            return self.fixture.output(task, progress)
        worker = threading.Thread(target=run_worker, args=(self.settings,), kwargs={'once':True,'runner':compute})
        worker.start()
        try:
            self.assertTrue(started.wait(5))
            stopped = self.client.post('/api/v1/jobs/' + fit['job_id'] + '/cancel').json()
            self.assertEqual(stopped['state'], 'cancelling')
            self.assertEqual(self.client.get('/api/v1/capabilities').json()['gpu']['available_count'], 0)
            session = self.client.get('/api/v1/sessions').json()[0]
            self.assertEqual(self.client.post('/api/v1/sessions/' + session['id'] + '/close').json()['status'], 'closing')
        finally:
            finish.set();worker.join(10)
        result = self.client.get('/api/v1/fits/' + fit['id']).json()
        self.assertEqual(result['state'], 'cancelled')
        self.assertFalse(result['artifacts_ready'])
        self.assertEqual(self.client.get('/api/v1/capabilities').json()['gpu']['available_count'], 1)
        self.assertEqual(self.client.get('/api/v1/sessions/' + session['id']).json()['session']['status'], 'closed')

    def test_completed_generation_survives_close_and_cancellation_cannot_undo_success(self):
        from backend.worker import run_worker
        from backend.tests.test_generations import GenerationApiTests
        from uuid import uuid4
        fit = self.client.post(self.fixture.prefix + '/fits', json=self.fixture.request()).json()
        run_worker(self.settings, once=True, runner=self.fixture.output)
        gen = self.client.post('/api/v1/fits/' + fit['id'] + '/generations', json={'request_id':str(uuid4()),'channel_id':'12.03','objective':'stretch','clip_count':1,'duration_seconds':1}).json()
        run_worker(self.settings, once=True, generation_runner=GenerationApiTests.compute)
        url = '/api/v1/generations/' + gen['id'] + '/download'
        saved = self.client.get(url).content
        self.assertEqual(self.client.post('/api/v1/jobs/' + gen['job_id'] + '/cancel').json()['state'], 'succeeded')
        session = self.client.get('/api/v1/sessions').json()[0]
        self.assertEqual(self.client.post('/api/v1/sessions/' + session['id'] + '/close').json()['status'], 'closed')
        restarted = TestClient(create_app(self.settings, lambda: 0));restarted.auth=('operator','test-password')
        self.assertEqual(restarted.get(url).content, saved)
        self.assertEqual(restarted.get('/api/v1/sessions/' + session['id']).json()['jobs'][0]['state'], 'succeeded')

    def test_retry_is_a_distinct_idempotent_attempt(self):
        from backend.worker import run_worker
        from uuid import uuid4
        fit = self.client.post(self.fixture.prefix + '/fits', json=self.fixture.request()).json()
        def fail(task, progress): raise RuntimeError('Controlled model failure')
        run_worker(self.settings, once=True, runner=fail)
        command = {'request_id':str(uuid4())}
        response = self.client.post('/api/v1/jobs/' + fit['job_id'] + '/retry', json=command)
        self.assertEqual(response.status_code, 202, response.text)
        retry = response.json();self.assertNotEqual(retry['id'], fit['id'])
        repeated = self.client.post('/api/v1/jobs/' + fit['job_id'] + '/retry', json=command)
        self.assertEqual(repeated.json()['id'], retry['id'])
        run_worker(self.settings, once=True, runner=self.fixture.output)
        self.assertEqual(self.client.get('/api/v1/fits/' + retry['id']).json()['state'], 'succeeded')
        self.assertEqual(self.client.get('/api/v1/fits/' + fit['id']).json()['state'], 'failed')
        detail = self.client.get('/api/v1/sessions/' + self.client.get('/api/v1/sessions').json()[0]['id']).json()
        self.assertEqual(next(job for job in detail['jobs'] if job['id']==retry['id'])['retry_of_job_id'],fit['job_id'])

    def test_close_is_scoped_and_authentication_required(self):
        first = self.client.post(self.fixture.prefix + '/fits', json=self.fixture.request()).json()
        session = self.client.get('/api/v1/sessions').json()[0]
        other = self.client.post('/api/v1/experiments', json={'gpu_count':1}).json()
        from uuid import uuid4
        prefix = '/api/v1/experiments/' + other['id']
        selected = self.client.post(prefix+'/recording-selection',json={'recording_id':self.fixture.recordings[0]['id']}).json()
        second = self.client.post(prefix+'/fits',json={'request_id':str(uuid4()),'recording_id':selected['recording']['id'],'recording_revision':1}).json()
        anonymous=TestClient(create_app(self.settings,lambda:1))
        for url in ['/api/v1/sessions', '/api/v1/sessions/'+session['id'],self.fixture.prefix+'/work']:
            self.assertEqual(anonymous.get(url).status_code,401)
        self.assertEqual(anonymous.post('/api/v1/jobs/'+first['job_id']+'/cancel').status_code,401)
        self.assertEqual(anonymous.post('/api/v1/sessions/'+session['id']+'/close').status_code,401)
        self.client.post('/api/v1/sessions/'+session['id']+'/close')
        self.assertEqual(self.client.get('/api/v1/fits/'+first['id']).json()['state'],'cancelled')
        self.assertEqual(self.client.get('/api/v1/fits/'+second['id']).json()['state'],'queued')
        self.assertEqual(self.client.post('/api/v1/experiments?session_id='+session['id'],json={'gpu_count':1}).status_code,409)
        self.assertEqual(self.client.post(self.fixture.prefix+'/recording-selection',json={'recording_id':self.fixture.recordings[1]['id']}).status_code,409)

    def test_close_terminates_real_child_and_preserves_completed_batch_targets(self):
        import os,json,threading,time,zipfile,io
        from pathlib import Path
        from uuid import uuid4
        from backend.worker import run_worker
        fit=self.client.post(self.fixture.prefix+'/fits',json=self.fixture.request()).json()
        run_worker(self.settings,once=True,runner=self.fixture.output)
        batch=self.client.post('/api/v1/fits/'+fit['id']+'/batches',json={'request_id':str(uuid4()),'gpu_count':1,'duration_seconds':1,'clip_count':1,'targets':[{'channel_id':'12.03','objective':'stretch'},{'channel_id':'12.03','objective':'one-hot'}]}).json()
        root=self.settings.database_path.parent
        fixture=root/'controlled-child';package=fixture/'backend';package.mkdir(parents=True)
        (package/'__init__.py').write_text('')
        (package/'generation_compute.py').write_text("""
import json,sys,time,wave
from pathlib import Path
root=Path(sys.argv[1]);job=json.loads((root/'input.json').read_text())['generation']
if job['objective']=='one-hot': time.sleep(30)
with wave.open(str(root/'clip-001.wav'),'wb') as audio:
    audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(16000);audio.writeframes(b'\\x01\\x00'*15392)
""")
        previous=Path.cwd();worker=None
        try:
            os.chdir(fixture)
            worker=threading.Thread(target=run_worker,args=(self.settings,),kwargs={'once':True,'gpu_count':lambda:1});worker.start()
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                current=self.client.get('/api/v1/batches/'+batch['id']).json()
                if current['targets'][0]['state']=='succeeded' and (root/'batches'/batch['id']/current['targets'][1]['id']/'process.json').exists():break
                time.sleep(.1)
            else:self.fail('Controlled child did not start')
            session=self.client.get('/api/v1/sessions').json()[0]
            self.client.post('/api/v1/sessions/'+session['id']+'/close')
            worker.join(10);self.assertFalse(worker.is_alive())
        finally:
            os.chdir(previous)
            if worker and worker.is_alive():worker.join(35)
        result=self.client.get('/api/v1/batches/'+batch['id']).json()
        self.assertEqual(result['state'],'cancelled')
        self.assertEqual([target['state'] for target in result['targets']],['succeeded','cancelled'])
        child=json.loads((root/'batches'/batch['id']/result['targets'][1]['id']/'process.json').read_text())
        self.assertLess(child['exit_code'],0)
        response=self.client.get('/api/v1/batches/'+batch['id']+'/download')
        self.assertEqual(response.status_code,200)
        self.assertEqual(len(zipfile.ZipFile(io.BytesIO(response.content)).namelist()),2)
        self.assertEqual(self.client.get('/api/v1/capabilities').json()['gpu']['available_count'],1)

    def test_batch_retry_only_repeats_unfinished_targets(self):
        from uuid import uuid4
        from backend.worker import run_worker
        from backend.tests.test_generations import GenerationApiTests
        fit=self.client.post(self.fixture.prefix+'/fits',json=self.fixture.request()).json()
        run_worker(self.settings,once=True,runner=self.fixture.output)
        body={'request_id':str(uuid4()),'gpu_count':1,'duration_seconds':1,'clip_count':1,'targets':[{'channel_id':'12.03','objective':objective} for objective in ['stretch','one-hot']]}
        batch=self.client.post('/api/v1/fits/'+fit['id']+'/batches',json=body).json()
        def compute(task,progress):
            if task.generation.objective=='one-hot':return 7
            return GenerationApiTests.compute(task,progress)
        run_worker(self.settings,once=True,generation_runner=compute,gpu_count=lambda:1)
        original=self.client.get('/api/v1/batches/'+batch['id']).json()
        self.assertEqual(original['state'],'partial')
        request={'request_id':str(uuid4())}
        response=self.client.post('/api/v1/jobs/'+batch['job_id']+'/retry',json=request)
        self.assertEqual(response.status_code,202,response.text)
        retry=response.json()
        self.assertNotEqual(retry['id'],batch['id'])
        self.assertEqual([target['objective'] for target in retry['targets']],['one-hot'])
        self.assertEqual(self.client.post('/api/v1/jobs/'+batch['job_id']+'/retry',json=request).json()['id'],retry['id'])
        run_worker(self.settings,once=True,generation_runner=GenerationApiTests.compute,gpu_count=lambda:1)
        self.assertEqual(self.client.get('/api/v1/batches/'+retry['id']).json()['state'],'succeeded')
        self.assertEqual(self.client.get('/api/v1/batches/'+batch['id']).json(),original)

    def test_retry_identity_cannot_be_reassigned_to_another_attempt(self):
        from uuid import uuid4
        from backend.worker import run_worker
        fit=self.client.post(self.fixture.prefix+'/fits',json=self.fixture.request()).json()
        run_worker(self.settings,once=True,runner=self.fixture.output)
        jobs=[]
        for _ in range(2):
            job=self.client.post('/api/v1/fits/'+fit['id']+'/generations',json={'request_id':str(uuid4()),'channel_id':'12.03','objective':'stretch'}).json()
            run_worker(self.settings,once=True,generation_runner=lambda task,progress:7)
            jobs.append(job)
        request={'request_id':str(uuid4())}
        self.assertEqual(self.client.post('/api/v1/jobs/'+jobs[0]['job_id']+'/retry',json=request).status_code,202)
        self.assertEqual(self.client.post('/api/v1/jobs/'+jobs[1]['job_id']+'/retry',json=request).status_code,409)

    def test_restart_waits_for_surviving_compute_child_before_releasing_gpu(self):
        import os,json,multiprocessing,time
        from pathlib import Path
        from uuid import uuid4
        from backend.worker import run_worker
        fit=self.client.post(self.fixture.prefix+'/fits',json=self.fixture.request()).json()
        run_worker(self.settings,once=True,runner=self.fixture.output)
        job=self.client.post('/api/v1/fits/'+fit['id']+'/generations',json={'request_id':str(uuid4()),'channel_id':'12.03','objective':'stretch'}).json()
        root=self.settings.database_path.parent;fixture=root/'surviving-child';package=fixture/'backend';package.mkdir(parents=True)
        (package/'__init__.py').write_text('')
        (package/'generation_compute.py').write_text('import time;time.sleep(4)')
        context=multiprocessing.get_context('fork');previous=Path.cwd();process=None
        try:
            os.chdir(fixture)
            process=context.Process(target=run_worker,args=(self.settings,),kwargs={'once':True});process.start()
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                if (root/'generations'/job['id']/'process.json').exists():break
                time.sleep(.05)
            else:self.fail('Owned child did not start')
            process.terminate();process.join(5)
            with self.assertRaisesRegex(RuntimeError,'already owns'):
                run_worker(self.settings,once=True)
            self.assertEqual(self.client.get('/api/v1/capabilities').json()['gpu']['available_count'],0)
            time.sleep(4.2)
            run_worker(self.settings,once=True)
        finally:
            os.chdir(previous)
            if process is not None and process.is_alive():process.terminate();process.join(5)
        self.assertEqual(self.client.get('/api/v1/generations/'+job['id']).json()['state'],'interrupted')
        self.assertEqual(self.client.get('/api/v1/capabilities').json()['gpu']['available_count'],1)

    def test_retry_cannot_adopt_an_existing_independent_attempt(self):
        from uuid import uuid4
        from backend.worker import run_worker
        bodies=[self.fixture.request(),self.fixture.request()]
        jobs=[]
        def fail(task,progress):raise RuntimeError('Controlled failure')
        for body in bodies:
            jobs.append(self.client.post(self.fixture.prefix+'/fits',json=body).json())
            run_worker(self.settings,once=True,runner=fail)
        response=self.client.post('/api/v1/jobs/'+jobs[0]['job_id']+'/retry',json={'request_id':bodies[1]['request_id']})
        self.assertEqual(response.status_code,409,response.text)
        detail=self.client.get('/api/v1/sessions/'+self.client.get('/api/v1/sessions').json()[0]['id']).json()
        self.assertEqual(len(detail['jobs']),2)
        self.assertTrue(all(job['retry_of_job_id'] is None for job in detail['jobs']))
        request={'request_id':str(uuid4())}
        retry=self.client.post('/api/v1/jobs/'+jobs[0]['job_id']+'/retry',json=request).json()
        body=self.fixture.request();body['request_id']=request['request_id']
        self.assertEqual(self.client.post(self.fixture.prefix+'/fits',json=body).status_code,409)
        self.assertNotIn(retry['id'],[job['id'] for job in jobs])

    def test_interrupted_batch_with_all_saved_targets_has_no_retry(self):
        import sqlite3,json
        from uuid import uuid4
        from backend.worker import run_worker
        from backend.tests.test_generations import GenerationApiTests
        fit=self.client.post(self.fixture.prefix+'/fits',json=self.fixture.request()).json()
        run_worker(self.settings,once=True,runner=self.fixture.output)
        batch=self.client.post('/api/v1/fits/'+fit['id']+'/batches',json={'request_id':str(uuid4()),'gpu_count':1,'duration_seconds':1,'clip_count':1,'targets':[{'channel_id':'12.03','objective':'stretch'}]}).json()
        run_worker(self.settings,once=True,generation_runner=GenerationApiTests.compute,gpu_count=lambda:1)
        # Persist the crash boundary: every target committed, parent finalization lost.
        with sqlite3.connect(str(self.settings.database_path)) as connection:
            data=json.loads(connection.execute('SELECT data FROM batches WHERE id=?',(batch['id'],)).fetchone()[0])
            data['state']='running';data['finished_at']=None
            connection.execute("UPDATE batches SET state='running',data=? WHERE id=?",(json.dumps(data),batch['id']))
        run_worker(self.settings,once=True,gpu_count=lambda:1)
        result=self.client.get('/api/v1/batches/'+batch['id']).json()
        self.assertEqual(result['state'],'interrupted')
        self.assertEqual(result['targets'][0]['state'],'succeeded')
        response=self.client.post('/api/v1/jobs/'+batch['job_id']+'/retry',json={'request_id':str(uuid4())})
        self.assertEqual(response.status_code,409,response.text)
        self.assertEqual(self.client.get('/api/v1/batches/'+batch['id']+'/download').status_code,200)
        detail=self.client.get('/api/v1/sessions/'+self.client.get('/api/v1/sessions').json()[0]['id']).json()
        self.assertFalse(next(job for job in detail['jobs'] if job['id']==batch['id'])['retryable'])


if __name__ == '__main__': unittest.main()
