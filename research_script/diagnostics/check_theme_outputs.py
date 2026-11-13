"""Read campaign progress and report missing/failed/completed theme units."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import OUTPUTS,THEMES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs-root',type=Path,default=OUTPUTS)
    args=p.parse_args();root=args.outputs_root
    progress=root/'00/campaign/progress.json'
    if progress.exists():
        d=json.loads(progress.read_text())
        print('Campaign:',d['status'],'PID:',d['pid'],'phase:',d.get('phase','preparing'))
    for number in ['02','03','04','05','06','07','08','09','10','12','13']:
        counts={};paths=list((root/number).glob('sub*/manifest.json'))
        if number=='10' and (root/number/'manifest.json').exists():paths=[root/number/'manifest.json']
        for path in paths:
            status=json.loads(path.read_text()).get('status','UNKNOWN');counts[status]=counts.get(status,0)+1
        failures=list((root/number).glob('sub*/failure.json'))
        print(number,THEMES[number],counts,'failure_reports=',len(failures))
    for path in sorted((root/'00/campaign/tasks').glob('*.json')):
        d=json.loads(path.read_text())
        print(path.stem,'theme',d['current_theme'],d['status'])
    panel_progress=root/'00/panel_campaign/progress.json'
    if panel_progress.exists():
        d=json.loads(panel_progress.read_text())
        print('COSYNE panels:',d['status'],'PID:',d['pid'],'phase:',d.get('phase','analysis'),
              'finished units:',len(d.get('tasks',{})))
        for number in ['01','02','03','04','05','09','12']:
            counts={}
            for path in (root/number/'panels').glob('sub*/manifest.json'):
                status=json.loads(path.read_text()).get('status','UNKNOWN');counts[status]=counts.get(status,0)+1
            print('panels',number,counts)
        print('panel failure reports:',len(list((root/'00/panel_campaign').glob('failure_*.json'))))

if __name__=='__main__':main()
