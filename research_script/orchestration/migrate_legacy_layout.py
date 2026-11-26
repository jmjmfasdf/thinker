"""One-time, guarded migration of completed results and obsolete entrypoints.

Default is read-only. --apply refuses to run while old research entrypoints or
output paths are used by any live process. It never terminates those processes,
changes training/capture data, or silently overwrites an existing archive.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'research_script'
ARCHIVE=REPO/'legacy/research_script/pre_reorganization'
OLD_ENTRIES=('00_training_diagnostics.py','01_model_fidelity.py','02_action_boundary_mechanisms.py',
 '03_dynamic_computation_control.py','04_search_to_action.py','05_human_prediction.py',
 '06_neural_decoding.py','07_neural_rsa.py','08_neural_encoding.py','09_task_state_complexity.py',
 '10_computation_utility.py','12_continuous_dynamics_analysis.py','13_directional_evaluation_accumulation.py',
 '13_1_bout_evaluation_dynamics.py','14_imagination_control_motifs.py')


# Only named research-result families are eligible. Unknown operational paths
# are preserved by default; this utility is not a general outputs cleanup.
LEGACY_OUTPUTS={*(f'{i:02d}' for i in range(1,15)), '13_1',
    '02_evaluation_revision_geometry_20260929_v10','03_dynamic_voc_mechanisms'}
LEGACY_ZERO_RESULTS={
    'planning_budget_comparison_20261113_650m','planning_candidate_analysis_20261114',
    'planning_learning_deep_review_20261113','planning_trend_20261113_6m'}
PROTECTED_DIR_PREFIXES=('00_dynamic_cache','visualization','training','checkpoint',
    'resume_','wandb_','voc10m_','behavioral_data','fmri')


def _canonical(path):
    """Also recognize group-only canonical roots, before any child is moved."""
    if not path.is_dir() or path.is_symlink():return False
    return (path/'logs/run.json').is_file() or any(path.glob('*/logs/run.json')) or any(path.glob('*/*/logs/run.json'))


def _protected_tree(path):
    """Never move a symlink or a tree containing links or operational data."""
    if path.is_symlink():return True
    if path.is_file():return path.suffix.lower() in ('.pt','.pth','.ckpt')
    if not path.is_dir():return True
    if path.name=='.work' or path.name.startswith(PROTECTED_DIR_PREFIXES):return True
    for directory,dirs,files in os.walk(path,followlinks=False):
        base=Path(directory)
        if any((base/name).is_symlink() or name=='.work' or name.startswith(PROTECTED_DIR_PREFIXES) for name in dirs):return True
        if any((base/name).is_symlink() or Path(name).suffix.lower() in ('.pt','.pth','.ckpt') for name in files):return True
    return False


def _touches(value,sources):
    try:target=Path(value.removesuffix(' (deleted)')).resolve()
    except (OSError,ValueError,RuntimeError):return False
    return any(target==source or source in target.parents for source in sources)


def live_dependencies(moves=None,proc_root=Path('/proc')):
    """Local process guard; remote scheduler state is deliberately not inferred.

    In addition to old entrypoint names, check candidate-path command arguments,
    current directories and open files. Unknown/remote operational directories
    stay outside the migration allowlist regardless of this local audit.
    """
    if moves is None:moves=plan()[0]
    sources=[Path(source).resolve() for source,_ in moves]
    blocked=[]
    for proc in proc_root.glob('[0-9]*'):
        if int(proc.name)==os.getpid():continue
        try:argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\x00') if x]
        except (OSError,PermissionError):continue
        if not argv:continue
        reason=None;reference=None
        if any(Path(arg).name in OLD_ENTRIES for arg in argv):reason='old_analysis_entrypoint'
        try:cwd=(proc/'cwd').resolve(strict=True)
        except (OSError,PermissionError):cwd=None
        if reason is None and cwd is not None and _touches(str(cwd),sources):reason='working_directory_in_move';reference=str(cwd)
        if reason is None:
            for arg in argv:
                value=arg.split('=',1)[-1] if arg.startswith('--') and '=' in arg else arg
                if '/' not in value:continue
                candidate=Path(value) if value.startswith('/') else cwd/value if cwd is not None else None
                if candidate is not None and _touches(str(candidate),sources):reason='argument_references_move';reference=value;break
        if reason is None:
            try:
                for fd in (proc/'fd').iterdir():
                    try:value=os.readlink(fd)
                    except (OSError,PermissionError):continue
                    if value.startswith('/') and _touches(value,sources):reason='open_file_in_move';reference=value;break
            except (OSError,PermissionError):pass
        if reason:
            # Do not persist arbitrary process arguments, which can contain keys.
            blocked.append(dict(pid=int(proc.name),executable=argv[0],reason=reason,reference=reference))
    return blocked


def plan():
    moves=[];preserved=[]
    out=ROOT/'outputs'
    if not out.is_dir():return moves,preserved
    def preserve(path):preserved.append(str(path))
    def add(path,destination):
        if _canonical(path) or _protected_tree(path):preserve(path)
        else:moves.append((path,destination))
    for p in sorted(out.iterdir()):
        if p.name=='00':continue
        if p.name not in LEGACY_OUTPUTS or p.is_symlink():preserve(p);continue
        destination=ARCHIVE/'outputs'/p.name
        if _canonical(p) or _protected_tree(p):
            # A current group unit alone is sufficient to protect the root.
            # Split only its immediate old siblings; preserve protected children
            # intact so their relative symlinks and metadata retain context.
            if p.is_dir() and not p.is_symlink() and not (p/'logs/run.json').is_file():
                for child in sorted(p.iterdir()):add(child,destination/child.name)
            else:preserve(p)
        else:moves.append((p,destination))
    zero=out/'00'
    if zero.is_symlink():preserve(zero)
    elif zero.is_dir():
        for p in sorted(zero.iterdir()):
            if p.name in LEGACY_ZERO_RESULTS:add(p,ARCHIVE/'outputs/00'/p.name)
            else:preserve(p)
    for name in OLD_ENTRIES:
        p=ROOT/name
        if p.is_symlink():preserve(p)
        elif p.is_file():moves.append((p,ARCHIVE/'retired_entrypoints'/name))
    # Preserving a link also means preserving any target that another proposed
    # move would relocate. This matters for relative links between siblings.
    targets=[]
    for directory,dirs,files in os.walk(out,followlinks=False):
        for name in dirs+files:
            link=Path(directory)/name
            if link.is_symlink():
                try:targets.append(link.resolve())
                except (OSError,RuntimeError):preserve(link)
    safe=[]
    for source,destination in moves:
        src=source.resolve()
        if any(src==target or src in target.parents or target in src.parents for target in targets):preserve(source)
        else:safe.append((source,destination))
    return safe,sorted(set(preserved))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');a=p.parse_args()
    moves,preserved=plan();blockers=live_dependencies(moves)
    report=dict(status='BLOCKED_ACTIVE_PROCESS' if blockers else 'READY',
        dependency_scope='Local processes only; operational, imported, native capture and unknown directories are preserved',
        blockers=blockers,moves=[dict(source=str(s),destination=str(d)) for s,d in moves],preserved=preserved)
    if not a.apply:
        print(json.dumps(report,indent=2));return
    if blockers:
        print(json.dumps(report,indent=2));raise SystemExit('Migration refused: wait for active old analyses to terminate')
    collision=[str(d) for s,d in moves if d.exists() or d.is_symlink()]
    if collision:raise FileExistsError('Archive already exists; inspect before retry: '+', '.join(collision))
    journal=REPO/'logs/maintenance/research_reorganization/migration.json'
    journal.parent.mkdir(parents=True,exist_ok=True)
    completed=[]
    for source,destination in moves:
        # Recheck immediately before every mutation; no race with a known old run.
        if live_dependencies(moves):raise RuntimeError('An old analysis started; stopped before next move')
        inode=source.lstat().st_ino;destination.parent.mkdir(parents=True,exist_ok=True)
        os.rename(source,destination)
        if destination.lstat().st_ino!=inode:raise RuntimeError('Rename inode mismatch')
        completed.append(dict(source=str(source),destination=str(destination),inode=inode))
        journal.write_text(json.dumps(dict(status='MIGRATING',moves=completed,preserved=preserved),indent=2)+'\n')
    report.update(status='COMPLETED',moves=completed)
    journal.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
