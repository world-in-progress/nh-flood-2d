"""Hydraulic connectivity, gate order and initial-inventory semantics."""
from types import SimpleNamespace as NS
import json

import numpy as np
import pytest

from src.nh_flood_2d.preprocess.hydraulics import connected_still_water


def boundary(name, cell, level):
    return NS(name=name, kind='level', cells=np.array([cell]), series=NS(at=lambda t: level))


def fill(bed, levels, *, crest=None, closed=None):
    # Exterior + chain of four cells. No route through exterior cell 0.
    sides = np.array([[0,0,0],[2,0,1],[2,1,2],[2,2,3],[2,3,4],[2,4,0]])
    return connected_still_water(np.array([0,*bed], dtype='float32'),
        np.full(6,-999.) if crest is None else np.array(crest),
        np.zeros(6, dtype='uint8') if closed is None else np.array(closed),
        sides, levels, np.array([0,1,1,1,1]))


def test_bed_barrier_is_strict_and_isolated_low_pool_stays_empty():
    depth, meta = fill([-2,-1,1,-3], [boundary('sea',1,1)])
    np.testing.assert_array_equal(depth,[0,3,2,0,0])
    assert meta['storage_m3'] == 5


@pytest.mark.parametrize('obstacle', ['crest','closed'])
def test_actual_edge_crest_and_closed_side_block_filling(obstacle):
    kw = {'crest':[-999,-999,-999,1,-999,-999]} if obstacle=='crest' else {'closed':[0,0,0,1,0,0]}
    depth,_ = fill([-2,-1,-2,-3], [boundary('sea',1,1)], **kw)
    np.testing.assert_array_equal(depth,[0,3,2,0,0])


@pytest.mark.parametrize('reverse',[False,True])
def test_connected_different_levels_reject_in_either_order(reverse):
    levels=[boundary('sea_A',1,1),boundary('sea_B',4,.5)]
    with pytest.raises(ValueError,match='Conflicting initial levels'):
        fill([-2,-1,-2,-3], levels[::-1] if reverse else levels)


def test_disconnected_levels_fill_separately_and_equal_levels_share_body():
    depth,_=fill([-2,2,-2,-3], [boundary('A',1,1),boundary('B',4,.5)])
    np.testing.assert_array_equal(depth,[0,3,0,2.5,3.5])
    depth,meta=fill([-2,-1,-2,-3], [boundary('A',1,1),boundary('B',4,1)])
    np.testing.assert_array_equal(depth,[0,3,2,3,4])
    assert len(meta['levels'])==1 and meta['levels'][0]['connected_components']==1


def test_initial_gate_state_changes_connectivity():
    shut,_=fill([-2,2,-2,-3], [boundary('sea',1,1)])
    opened,_=fill([-2,0,-2,-3], [boundary('sea',1,1)])
    assert shut[3]==0 and opened[3]==3


@pytest.mark.usefixtures('taichi_cpu')
def test_warmup_fills_connected_cells_into_initial_inventory(tmp_path):
    from test_solver_forcing import experiment
    result,_,meta=experiment(tmp_path,[[0,0]],1,connected=True,level=1,warmup=1,warmup_only=True)
    assert meta['initialization']['method']=='connected_still_water'
    assert result['initial_storage_m3']==200
    assert result['level:sea_m3']==0 and result['residual_m3']==0
    from src.nh_flood_2d.core.output import read_checkpoint
    state=read_checkpoint(tmp_path/'out/output/state_final.fdb')
    np.testing.assert_array_equal(state.depth,[0,1,1])
    assert not np.any(state.q)


@pytest.mark.usefixtures('taichi_cpu')
def test_no_warmup_preserves_dry_initialization(tmp_path):
    from test_solver_forcing import experiment
    result,_,meta=experiment(tmp_path,[[0,0]],1,connected=True,level=1)
    assert meta['initialization']['method']=='dry'
    assert result['initial_storage_m3']==0


@pytest.mark.usefixtures('taichi_cpu')
def test_checkpoint_skips_automatic_fill_even_with_warmup(tmp_path):
    from test_restart import _inputs,START
    from src.nh_flood_2d.input import DomainConfig,ForceConfig
    from src.nh_flood_2d.core.output import write_checkpoint
    from src.nh_flood_2d.core.run_record import sha256
    from src.nh_flood_2d.core.solver_compact import warmup_solver
    from src.nh_flood_2d.preprocess import preprocess
    inputs=tmp_path/'inputs';_inputs(inputs)
    (inputs/'sea.geojson').write_text(json.dumps(dict(type='LineString',coordinates=[[0,1],[0,9]],crs=dict(type='name',properties=dict(name='EPSG:2326')))))
    (inputs/'sea.csv').write_text('time,value\n2023-09-07T08:00:00+00:00,1\n2023-09-07T08:00:10+00:00,1\n')
    seed=inputs/'seed.fdb';depth=np.array([0,.25,.25],dtype='float32');zeros=np.zeros(3,dtype='float32');words=np.zeros(3,dtype='uint32')
    write_checkpoint(seed,depth=depth,bed=zeros,q=np.zeros(8,dtype='float32'),rain_time=zeros,infiltration=zeros,max_depth=depth,max_speed=zeros,first_time_hi=words,first_time_lo=words,clock_hi=0,clock_lo=0,gate_open=False,metadata_json=json.dumps(dict(run_state=False,ne_sha256=sha256(inputs/'ne.txt'),ns_sha256=sha256(inputs/'ns.txt'),gate_config='null')))
    dc=DomainConfig(ne=str(inputs/'ne.txt'),ns=str(inputs/'ns.txt'),domain_dir=str(tmp_path/'out'),epsg_code=2326,start=START,duration=1,warmup_duration=1,land_parameters={10:dict(manning_n=.03,infiltration='none')},restart_state=str(seed))
    fc=ForceConfig(boundaries=[dict(name='sea',kind='level',line=str(inputs/'sea.geojson'),series=str(inputs/'sea.csv'))],rainfall=dict(raster=str(inputs/'rain.tif'),intervals=str(inputs/'intervals.csv')))
    preprocess(dc,fc);result=warmup_solver(dc,fc)
    meta=json.loads((tmp_path/'out/manifest.json').read_text())
    assert meta['initialization']['method']=='checkpoint'
    assert result['initial_storage_m3']==50


@pytest.mark.usefixtures('taichi_cpu')
def test_gate_open_at_start_precedes_fill_and_is_initial_inventory(tmp_path):
    from test_solver_forcing import experiment
    result,_,meta=experiment(tmp_path,[[0,0]],1,land_type=11,bed=2,
        connected=True,level=1,warmup=1,warmup_only=True,
        gate=dict(threshold=.5,land_type=11,level_boundary='sea'))
    assert meta['gate_open'] and meta['bed_events'][0]['elapsed_seconds']==-1
    assert meta['initialization']['wet_cells']==2
    assert result['initial_storage_m3']==200
    assert result['level:sea_m3']==0 and result['gate_bed_m3']==0


def test_connectivity_compares_input_level_without_rounding_it_to_f32():
    # A level less than half an F32 ulp above the bed is still strictly higher.
    depth,_=fill([1,1,1,1], [boundary('sea',1,1+1e-8)])
    np.testing.assert_allclose(depth[1:],1e-8,rtol=1e-7)
