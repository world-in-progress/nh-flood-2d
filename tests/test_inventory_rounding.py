"""Tiny rain and a transport increment must meet before the F32 inventory."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.nh_flood_2d.core.output import read_checkpoint, write_checkpoint
from src.nh_flood_2d.core.run_record import sha256
from src.nh_flood_2d.core.solver_compact import run_simulation
from src.nh_flood_2d.input import DomainConfig, ForceConfig

pytestmark = pytest.mark.usefixtures('taichi_cpu')


def test_transport_and_small_rain_have_one_inventory_rounding(tmp_path):
    start = datetime(2023, 9, 7, 8, tzinfo=timezone.utc)
    (tmp_path/'ne.txt').write_text(
        '1,1,1,1,1,1,2,3,4,5,5,0,10\n'
        '2,1,1,1,1,2,5,6,7,15,5,0,10\n')
    (tmp_path/'ns.txt').write_text(
        '1,2,0,1,0,0,10,0,5,0,0\n2,2,1,2,0,0,10,10,5,0,0\n'
        '3,1,0,0,0,1,10,5,0,0,0\n4,1,0,0,1,0,10,5,10,0,0\n'
        '5,2,2,0,0,0,10,20,5,0,0\n6,1,0,0,0,2,10,15,0,0,0\n'
        '7,1,0,0,2,0,10,15,10,0,0\n')
    mm = np.float32(1.4911491234670393e-5)
    with rasterio.open(tmp_path/'rain.tif','w',driver='GTiff',width=2,height=1,count=1,
                       dtype='float32',crs='EPSG:2326',transform=from_origin(0,10,10,10)) as dst:
        dst.write(np.array([[mm,0]],dtype='float32'),1)
    (tmp_path/'times.csv').write_text('interval_start,interval_end\n'+
        f'{start.isoformat()},{(start+timedelta(seconds=1)).isoformat()}\n')
    depth = np.array([0,.5,.4],dtype='float32');zeros=np.zeros(3,dtype='float32')
    words=np.zeros(3,dtype='uint32')
    write_checkpoint(tmp_path/'seed.fdb',depth=depth,bed=zeros,q=np.zeros(8,dtype='float32'),
        rain_time=zeros,infiltration=zeros,clock_hi=0,clock_lo=0,gate_open=False,
        max_depth=depth,first_time_hi=words,first_time_lo=words,max_speed=zeros,
        metadata_json=json.dumps(dict(run_state=False,ne_sha256=sha256(tmp_path/'ne.txt'),
            ns_sha256=sha256(tmp_path/'ns.txt'),gate_config='null')))
    dc=DomainConfig(ne=str(tmp_path/'ne.txt'),ns=str(tmp_path/'ns.txt'),epsg_code=2326,
        domain_dir=str(tmp_path/'out'),start=start,duration=1,warmup_duration=0,
        land_parameters={10:dict(manning_n=.03,infiltration='none')},restart_state=str(tmp_path/'seed.fdb'))
    fc=ForceConfig(boundaries=[],rainfall=dict(raster=str(tmp_path/'rain.tif'),intervals=str(tmp_path/'times.csv')))
    run_simulation(dc,fc)
    state=read_checkpoint(Path(dc.output_dir)/'state_final.fdb')
    assert json.loads((tmp_path/'out/manifest.json').read_text())['steps']==1
    # Use the actual shared edge flux in an independent F64 cell balance. The
    # chosen rain exposes an extra inventory rounding by almost a full ULP.
    rain_rate=float(np.float32(float(mm)*.001))
    ideal=float(depth[1])-float(state.q[2])*10/100+rain_rate
    ulp=float(np.spacing(np.float32(ideal)))
    assert abs(float(state.depth[1])-ideal) < .55*ulp
