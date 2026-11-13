"""Check delivered visual descriptors against sampled original human images."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
import sys
import tempfile
import zipfile
import shutil
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'thinker'))
from common.paths import REPO, OUTPUTS, INPUT_CACHE, tag
from dynamic_analysis_cache import visual_descriptor
from thinker.bc_loader import FrameStackedBehavioralDataLoader


def main(output):
    records=[]
    loader=object.__new__(FrameStackedBehavioralDataLoader)
    loader.target_size=(84,84);loader.grayscale=False;loader.normalize=False
    for game in range(3):
        folder=INPUT_CACHE/'checkpoint_leave01'/tag(1,game)
        m=json.loads((folder/'manifest.json').read_text())
        name,item=next((n,x) for n,x in m['source_runs'].items() if 'ses-01' in n)
        run=item['run_label'];path=REPO/'behavioral_data_block'/name
        align=pd.read_csv(folder/'visual_alignment.csv.gz')
        z=align.loc[align.run_label.eq(run)&align.previous_frame.ge(0)].copy()
        selected=z.iloc[np.linspace(0,len(z)-1,32).astype(int)]
        with tempfile.TemporaryDirectory(prefix='thinker-visual-audit-') as temp:
            extracted=Path(temp)/'image.npy'
            with zipfile.ZipFile(path) as archive,archive.open('image.npy') as src,extracted.open('wb') as dst:
                shutil.copyfileobj(src,dst,1<<20)
            images=np.load(extracted,mmap_mode='r')
            # Recorder materializes C-contiguous CHW uint8 images. Matching
            # layout also avoids float32 reduction-order differences.
            def get(ix):return np.ascontiguousarray(np.stack([loader._preprocess_frame(images[int(i)]) for i in ix]))
            current=get(selected.current_frame);past=get(selected.previous_frame)
            recomputed=visual_descriptor(current,past)
            with np.load(folder/'visual_controls.npz') as v:
                stored=v['descriptor'][selected.actor_row.to_numpy(int)]
            difference=float(np.max(abs(recomputed-stored)))
            assert difference<1e-7, (game,difference)
            pairs=pd.read_csv(OUTPUTS/'02/panels'/tag(1,game)/'tables/matched_pairs.csv')
            pairs=pairs.loc[pairs.run_label.eq(run)&pairs.visual_distance.eq(0)&pairs.population.eq('overall')].head(32)
            r=z.set_index('real_step_id')
            p=r.loc[pairs.switch_id];q=r.loc[pairs.persist_id]
            same=np.all(get(p.current_frame)==get(q.current_frame),axis=(1,2,3)) if len(pairs) else np.array([])
            records.append(dict(game=game,subject=1,checkpoint=1,run_label=run,n_samples=len(selected),
                descriptor_max_error=difference,n_zero_distance_pairs=len(pairs),
                identical_current_image_fraction=float(same.mean()) if len(same) else None))
            del images
    pd.DataFrame(records).to_csv(output/'visual_source_samples.csv',index=False)
    (output/'visual_source_audit.py').write_text(Path(__file__).read_text())
    print(pd.DataFrame(records).to_string(index=False))


if __name__=='__main__':main(Path(sys.argv[1]))
