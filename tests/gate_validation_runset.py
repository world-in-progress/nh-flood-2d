"""Run the explicitly authorized evening Metal checks sequentially.

The wait runs in a background process so the agent can keep responding. It
checks no command lines and never initializes Metal before the chosen window.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from gate_gpu_tools import gpu_window


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--start-at',required=True)
    parser.add_argument('--idle-confirmations',type=int,default=6)
    args=parser.parse_args()
    start=datetime.fromisoformat(args.start_at)
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    idle = 0
    while datetime.now(ZoneInfo('Asia/Singapore')) < start:
        state=gpu_window()
        idle = idle+1 if not state['solver_pids'] else 0
        (root/'waiting.json').write_text(json.dumps(state,indent=2)+'\n')
        print('WAITING '+state['time']+' solver_pids='+str(state['solver_pids']),flush=True)
        if idle >= args.idle_confirmations:
            print('CONFIRMED_IDLE '+state['time'],flush=True)
            break
        remaining=(start-datetime.now(ZoneInfo('Asia/Singapore'))).total_seconds()
        time.sleep(max(.1,min(50.,remaining)))
    script=Path(__file__).parent
    python=[sys.executable,'-B']
    jobs=[]
    for case in ('small','medium'):
        jobs.append((f'enabled-{case}',python+[str(script/'gate_backend_validation.py'),
            '--backend','metal','--case',case,'--output',str(root/f'enabled-{case}')]))
    jobs.append(('metal-pytest',python+['-m','pytest','-q','-p','no:cacheprovider',
        '--basetemp',str(root/'metal-pytest'),str(script/'test_gate_metal.py'),'--tb=short']))
    jobs.append(('compat-small',python+[str(script/'gate_compatibility.py'),
        '--backend','metal','--case','small','--reference-repeats','2',
        '--output',str(root/'compat-small')]))
    jobs.append(('compat-hk',python+[str(script/'gate_compatibility.py'),
        '--backend','metal','--case','hk','--reference-repeats','2',
        '--domain','<case-dir>/runs_formal/Bnone-I/config/domain.json',
        '--force','<case-dir>/runs_formal/Bnone-I/config/force.json',
        '--output',str(root/'compat-hk')]))
    for name,disabled in [('performance-disabled',True),('performance-enabled',False),
                           ('performance-disabled-repeat',True)]:
        command=python+[str(script/'gate_backend_validation.py'),'--backend','metal',
            '--case','medium','--performance','--output',str(root/name)]
        if disabled: command.append('--disabled')
        jobs.append((name,command))
    environment=os.environ.copy()
    environment.update(TI_OFFLINE_CACHE='0',PYTHONDONTWRITEBYTECODE='1',NH_GATE_METAL_TESTS='1')
    results=[]
    for name,command in jobs:
        window=gpu_window()
        if not window['allowed']:
            results.append(dict(name=name,not_started=True,window=window))
            break
        print('START '+name+' '+window['time'],flush=True)
        began=time.perf_counter()
        with (root/f'{name}.log').open('w') as stream:
            completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
                                     env=environment,check=False)
        result=dict(name=name,exit_code=completed.returncode,
                    wall_seconds=time.perf_counter()-began,window=window)
        results.append(result)
        (root/'runset.json').write_text(json.dumps(results,indent=2)+'\n')
        print('FINISH '+name+' exit='+str(completed.returncode),flush=True)
    (root/'runset.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(result.get('exit_code') != 0 for result in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
