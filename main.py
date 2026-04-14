from pathlib import Path

from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.preprocess.pipe import prepare_pipe
from src.nh_flood_2d.preprocess.warmstart import clean_uvh_for_warmstart
from src.nh_flood_2d.core.solver_compact import solver, warmup_solver
from src.nh_flood_2d.core.coupled import solver_coupled
from src.nh_flood_2d.output.hydrograph import draw_hydrograph, compare_hydrograph
from src.nh_flood_2d.input import load_domain_config, DomainConfig, load_force_config, ForceConfig
from src.nh_flood_2d.input.pipe import load_pipe_config, PipeConfig
from src.nh_flood_2d.output.flood_map import generate_flood_map, generate_max_inundation_extent_map, plot_spatial_mae_curve, generate_flood_video

df7_cfg = load_force_config('./resource/df7.json')
pipe_cfg   = load_pipe_config('./resource/pipe.json')
df11_cfg = load_force_config('./resource/df11.json')
domain_4 = load_domain_config('./resource/domain_4.json')
# domain_mrcg = load_domain_config('./resource/domain_mrcg.json')
domain_alt = load_domain_config('./resource/domain_alt.json')
# domain_mrcg_gw = load_domain_config('./resource/domain_mrcg_gw.json')
# domain_basic = load_domain_config('./resource/domain_basic.json')

def evolve_domain(domain_cfg: DomainConfig, force_cfg: ForceConfig):
    preprocess(domain_cfg, force_cfg)
    # warmup_solver(domain_cfg, force_cfg)
    solver(domain_cfg, force_cfg)

def evolve_domain_coupled(
    domain_cfg: DomainConfig,
    force_cfg: ForceConfig,
    pipe_cfg: PipeConfig | None = None,
    start_time_step: int = 0,
):
    """Preprocess domain & force data, optionally preprocess pipe data, then run solver.
    
    When pipe_cfg is None, runs as a 2D-only solver (equivalent to evolve_domain
    but uses the solver_coupled code path).
    """
    preprocess(domain_cfg, force_cfg)
    if pipe_cfg is not None:
        prepare_pipe(pipe_cfg, domain_cfg)
    solver_coupled(domain_cfg, force_cfg, pipe_cfg, start_time_step)

def evolve_domain_coupled_warmup(
    domain_cfg: DomainConfig,
    force_cfg: ForceConfig,
    pipe_cfg: PipeConfig | None = None,
    keep_types: tuple[int, ...] = (7, 8),
):
    """Run a coupled simulation, then clean the last UVH snapshot for warm-start.

    1. Calls evolve_domain_coupled (preprocess + solve).
    2. Finds the last UVH .fdb in domain_cfg.uvh_dir (by timestamp).
    3. Cleans it (keeps water only on *keep_types* elements) and writes
       a ``warmstart.fdb`` next to the UVH directory.

    The resulting file can be used as ``domain_cfg.restart_uvh`` in a
    subsequent cold→warm restart run.

    Parameters
    ----------
    keep_types : Element types that retain water (default 7=fish pond, 8=water body).
    """
    evolve_domain_coupled(domain_cfg, force_cfg, pipe_cfg)

    # Find the last UVH snapshot (sorted by filename timestamp)
    uvh_dir = Path(domain_cfg.uvh_dir)
    uvh_files = sorted(uvh_dir.glob('uvh_*.fdb'))
    if not uvh_files:
        print('[warmstart] No UVH files found — skipping warm-start cleanup.')
        return

    last_uvh = str(uvh_files[-1])
    warmstart_out = str(uvh_dir.parent / 'warmstart.fdb')

    print(f'[warmstart] Using last UVH: {last_uvh}')
    clean_uvh_for_warmstart(
        uvh_path=last_uvh,
        ne_fdb_path=domain_cfg.ne_fdb,
        output_path=warmstart_out,
        keep_types=keep_types,
    )
    print(f'[warmstart] Set restart_uvh to "{warmstart_out}" in your domain config to use it.')

if __name__ == '__main__':
    # Warmup
    # evolve_domain_coupled_warmup(domain_4, df11_cfg, pipe_cfg, keep_types=(7, 8))
    
    # Simulation
    # evolve_domain_coupled(domain_4, df7_cfg, pipe_cfg)
    
    # Output video
    generate_flood_map(domain_alt)
    generate_flood_video(domain_alt, output_path='./resource/flood_video_mrcg.mp4')
    
    # draw_hydrograph(domain_alt, 'D74', True, -3600)
    
    # preprocess(domain_mrcg, df7_cfg)
    # generate_max_inundation_extent_map(domain_4)
    
    # mses = compare_hydrograph([domain_4, domain_alt], 'D43C', show=True, show_obs=True, baseline=domain_4)
    # print(f'RMSEs: {mses}')
    
    # plot_spatial_mae_curve(domain_4, domain_mrcg, df7_cfg, output_path='./resource/spatial_mae_curve.png')
    