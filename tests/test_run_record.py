"""Source identity survives relocation and does not require Git metadata."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from src.nh_flood_2d.core.run_record import source_tree_sha256


def test_hash_recipe_sorts_relative_paths_and_includes_names_and_bytes(tmp_path):
    src=tmp_path/'src';(src/'pkg').mkdir(parents=True)
    (src/'pkg/z.py').write_bytes(b'z=1\n');(src/'a.py').write_bytes(b'a=2\n')
    (src/'ignored.txt').write_text('irrelevant')
    raw=b'nh-flood-2d/src-py-sha256/v1\0'
    for name in ['a.py','pkg/z.py']:
        body=(src/name).read_bytes();raw+=name.encode()+b'\0'+len(body).to_bytes(8,'big')+body
    expected=hashlib.sha256(raw).hexdigest()
    assert source_tree_sha256(src)==expected
    copied=tmp_path/'copy';shutil.copytree(src,copied)
    assert source_tree_sha256(copied)==expected
    (copied/'ignored.txt').write_text('changed');assert source_tree_sha256(copied)==expected
    (copied/'a.py').rename(copied/'b.py');assert source_tree_sha256(copied)!=expected
    (src/'a.py').write_bytes(b'a=3\n');assert source_tree_sha256(src)!=expected


def test_complete_simulation_in_copy_without_git(tmp_path):
    from test_restart import _inputs,START
    from src.nh_flood_2d.input import DomainConfig,ForceConfig
    source=Path(__file__).resolve().parents[1]
    copied=tmp_path/'distribution';copied.mkdir()
    shutil.copytree(source/'src',copied/'src',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(source/'main.py',copied/'main.py')
    inputs=tmp_path/'inputs';_inputs(inputs)
    dc=DomainConfig(ne=str(inputs/'ne.txt'),ns=str(inputs/'ns.txt'),epsg_code=2326,domain_dir=str(tmp_path/'out'),start=START,duration=1,warmup_duration=0,land_parameters={10:dict(manning_n=.03,infiltration='none')})
    fc=ForceConfig(boundaries=[],rainfall=dict(raster=str(inputs/'rain.tif'),intervals=str(inputs/'intervals.csv')))
    (copied/'domain.json').write_text(dc.model_dump_json());(copied/'force.json').write_text(fc.model_dump_json())
    code="from src.nh_flood_2d.util.ti import init_taichi; init_taichi(use_gpu=False); import runpy,sys; sys.argv=['main.py','--domain','domain.json','--force','force.json'];runpy.run_path('main.py',run_name='__main__')"
    run=subprocess.run([sys.executable,'-c',code],cwd=copied,text=True,capture_output=True)
    assert run.returncode==0,run.stdout+run.stderr
    meta=json.loads((tmp_path/'out/manifest.json').read_text())
    assert meta['status']=='completed' and meta['steps']>0
    assert meta['git_commit'] is None and meta['git_dirty'] is None
    assert meta['source_sha256']==source_tree_sha256(copied/'src')
