"""Archive pilot display artifacts, rebuild tables/rendering without any fitting."""
import hashlib
import json
import shutil
from pathlib import Path

from core import Context, parser
from panels import build
from render import render
from run import verify


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    args=parser().parse_args();ctx=Context(args)
    if args.subject!=1:raise ValueError('Pilot-only refresh')
    ctx.verify_seal(ctx.root/'derived');ctx.verify_seal(ctx.root/'fits')
    immutable={str(p.relative_to(ctx.root)):digest(p)
        for folder in ('derived','fits','provenance')
        for p in (ctx.root/folder).rglob('*') if p.is_file() and (folder!='provenance' or p.suffix=='.csv')}
    archive=ctx.root/'review_archive/20260922_pre_review'
    if archive.exists():raise FileExistsError('Prior review archive exists; do not overwrite')
    old=json.loads((ctx.root/'panels/manifest.json').read_text())
    for name,expected in old['artifacts'].items():
        if digest(ctx.root/'panels'/name)!=expected:raise ValueError('Changed original panel: '+name)
    archive.mkdir(parents=True)
    for name in ('panels','figures','completion.json'):
        shutil.move(str(ctx.root/name),str(archive/name))
    build(ctx);render(ctx);verify(ctx)
    for name,expected in immutable.items():
        if digest(ctx.root/name)!=expected:raise ValueError('Immutable numeric artifact changed: '+name)
    ctx.C.write_json(ctx.root/'review_refresh.json',dict(status='COMPLETED',
        archived=str(archive),unchanged_numeric_files=immutable,
        modification='Descriptive KM float ties and rendering only; no refit or retest'))
    print('Review refresh complete; unchanged numeric artifacts:',len(immutable),flush=True)


if __name__=='__main__':main()
