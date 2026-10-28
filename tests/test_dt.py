import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import httpx
import numpy as np
import torch
from dt.codec import pack, unpack, sha
from dt.coordinator import one_round
from dt.planner import plans
from src.fl.fedavg_core import model, state, ClientWorker, aggregate


class DTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.headers = {'X-DT-Token':'test-token-for-local-only'}
        cls.procs, cls.logs, cls.workers = [], [], []
        cls.case = {'clients': [], 'target': 4, 'rounds': 2, 'seed': 900042,
                    'input_dim': 3, 'hidden': [4,3],
                    'fl': {'lr':.001,'batch_size':8,'local_epochs':1}}
        try:
            for i in range(5):
                dest=cls.root/f'client{i}'
                dest.mkdir()
                rng=np.random.default_rng(i)
                x=rng.normal(size=(20+i*4,3)).astype('float32')
                y=np.arange(len(x))%2
                rows=np.array([f'{i}-{j}' for j in range(len(x))])
                np.savez(dest/'local.npz',X=x,y=y,row_id=rows)
                cls.workers.append(ClientWorker(i,x,y,rows))
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
                env=os.environ.copy()
                env.update({'CLIENT_ID':str(i),'DATA_DIR':str(dest),'CACHE_DIR':str(dest/'cache'),
                            'DT_TOKEN':'test-token-for-local-only','OMP_NUM_THREADS':'1'})
                log=(dest/'server.log').open('w')
                cls.logs.append(log)
                cls.procs.append(subprocess.Popen([sys.executable,'-m','uvicorn','dt.worker:app',
                    '--host','127.0.0.1','--port',str(port)],env=env,stdout=log,stderr=log,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
                cls.case['clients'].append({'id':i,'url':f'http://127.0.0.1:{port}','n':len(y),
                    'hist':np.bincount(y,minlength=2).tolist(), 'data_version':sha((dest/'local.npz').read_bytes())})
            deadline=time.time()+60
            for c in cls.case['clients']:
                while True:
                    try:
                        r=httpx.get(c['url']+'/health',timeout=1,trust_env=False)
                        r.raise_for_status(); break
                    except Exception:
                        if time.time()>deadline: raise RuntimeError('Server startup failed')
                        time.sleep(.2)
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        for proc in cls.procs:
            proc.terminate()
            proc.wait(timeout=15)
        for log in cls.logs: log.close()
        cls.tmp.cleanup()

    def test_http_round_matches_existing_weighted_fedavg(self):
        weights=state(model(3,[4,3],42))
        actual, log=one_round(weights,self.case,[0,1,2,3,4],'integration',0,self.headers)
        updates, counts=[],[]
        for w in self.workers:
            m=model(3,[4,3],0);m.load_state_dict(weights)
            u,n,_=w.train(m,self.case['fl'],900042+w.client_id*100)
            updates.append(u);counts.append(n)
        expected=aggregate(updates,counts)
        for k in actual: torch.testing.assert_close(actual[k],expected[k],rtol=0,atol=0)
        self.assertEqual(len(log['clients']),5)
        self.assertTrue(all(c['body_bytes']>0 for c in log['clients']))

    def test_cached_job_is_idempotent_and_rejects_collision(self):
        c=self.case['clients'][0]
        meta={'job_id':'retry_test','base_hash':'base','data_version':c['data_version'],
              'input_dim':3,'hidden':[4,3],'fl':self.case['fl'],'seed':42}
        payload=pack(state(model(3,[4,3],42)),meta)
        r=httpx.post(c['url']+'/train',content=payload,headers=self.headers,trust_env=False,timeout=30)
        r.raise_for_status()
        again=httpx.post(c['url']+'/train',content=payload,headers=self.headers,trust_env=False)
        self.assertEqual(r.content,again.content)
        meta['seed']=43
        collision=httpx.post(c['url']+'/train',content=pack(state(model(3,[4,3],42)),meta),headers=self.headers,trust_env=False)
        self.assertEqual(collision.status_code,409)
        bad=httpx.get(c['url']+'/state',trust_env=False)
        self.assertEqual(bad.status_code,401)

    def test_planner_no_feasible_stale_and_target(self):
        twin={'round_overhead_seconds':.01,'clients':{str(i):{'received_at':time.time(),
              'elapsed_seconds':.1+i*.1,'body_bytes':100} for i in range(5)}}
        valid=plans(self.case,twin,10,100000)
        self.assertTrue(valid)
        self.assertTrue(all(4 in v['clients'] for v in valid))
        self.assertEqual(plans(self.case,twin,10,1),[])
        twin['clients']['4']['received_at']=0
        self.assertEqual(plans(self.case,twin,10,100000),[])

    def test_export_retained_shards_and_central_boundary(self):
        from dt.export_case import export
        data=self.root/'export-data';data.mkdir()
        source=self.root/'source-run';(source/'checkpoints').mkdir(parents=True)
        hashes={}
        for split in ['train','validation','calibration','test']:
            n=100 if split=='train' else 10
            np.savez(data/f'{split}.npz',X=np.ones((n,3),dtype='float32'), y=np.arange(n)%2,
                     row_id=np.array([f'{split}-{i}' for i in range(n)]),
                     group_id=np.array([f'{split}-g{i}' for i in range(n)]))
            hashes[f'{split}.npz']=sha((data/f'{split}.npz').read_bytes())
        (data/'manifest.json').write_text(json.dumps({'artifacts_sha256':hashes}))
        (data/'audit.json').write_text(json.dumps({'training_ready':True,'cross_split_encoded_overlap':{'train/test':0}}))
        cfg={'clients':5,'fl':self.case['fl'],'hidden':[4,3],'seed':42,'recovery_rounds':2}
        (source/'manifest.json').write_text(json.dumps({'config':cfg}))
        np.savez(source/'partitions.npz',**{f'client_{i}':np.array([f'train-{j}' for j in range(i*20,(i+1)*20)]) for i in range(5)})
        (source/'request_0.1.json').write_text(json.dumps({'target':0,'forget_row_ids':['train-0','train-1']}))
        for name in ['original','0.1_after_local','0.1_after_kd']:
            torch.save({'state_dict':state(model(3,[4,3],42))},source/'checkpoints'/f'{name}.pt')
        dest=self.root/'exported'
        export(data,source,'0.1',dest)
        case=json.loads((dest/'central'/'case.json').read_text())
        self.assertEqual(sum(c['n'] for c in case['clients']),98)
        self.assertEqual(case['clients'][0]['n'],18)
        for i in range(5):
            with np.load(dest/f'client{i}'/'local.npz',allow_pickle=False) as z:
                self.assertFalse(set(z['row_id']) & {'train-0','train-1'})
        self.assertNotIn('row_id',(dest/'central'/'case.json').read_text())

    def test_cli_calibrate_run_and_no_feasible(self):
        central=self.root/'central';central.mkdir(exist_ok=True)
        case=central/'case.json';case.write_text(json.dumps(self.case))
        initial=central/'initial.npz';initial.write_bytes(pack(state(model(3,[4,3],42)),{}))
        twin=central/'twin.json'
        env=os.environ.copy();env['DT_TOKEN']='test-token-for-local-only'
        def run(mode,name,extra):
            return subprocess.run([sys.executable,'-m','dt.coordinator',mode,'--case',str(case),
                '--initial',str(initial),'--twin',str(twin),'--output',str(central/name),*extra],
                env=env,capture_output=True,text=True,timeout=90)
        r=run('calibrate','calibration',[])
        self.assertEqual(r.returncode,0,r.stderr)
        r=run('run','run',['--policy','all'])
        self.assertEqual(r.returncode,0,r.stderr)
        report=json.loads((central/'run'/'report.json').read_text())
        self.assertEqual(report['status'],'COMPLETED')
        self.assertEqual(len(report['logs']),2)
        r=run('run','adaptive',[])
        self.assertEqual(r.returncode,0,r.stderr)
        adaptive=json.loads((central/'adaptive'/'report.json').read_text())
        self.assertEqual(adaptive['status'],'COMPLETED')
        self.assertIn(4,adaptive['plan']['clients'])
        r=run('run','no-plan',['--budget-bytes','1'])
        self.assertEqual(r.returncode,0,r.stderr)
        report=json.loads((central/'no-plan'/'report.json').read_text())
        self.assertEqual(report['status'],'NO_FEASIBLE_PLAN')
        self.assertFalse((central/'no-plan'/'recovered.npz').exists())


if __name__=='__main__': unittest.main()
