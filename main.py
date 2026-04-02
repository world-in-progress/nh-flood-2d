from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.core.solver_compact import solver, warmup_solver
from src.nh_flood_2d.output.hydrograph import draw_hydrograph, compare_hydrograph
from src.nh_flood_2d.input import load_domain_config, DomainConfig, load_force_config, ForceConfig
from src.nh_flood_2d.output.flood_map import generate_flood_map, generate_max_inundation_extent_map, plot_spatial_mae_curve, generate_flood_video

df7_cfg = load_force_config('./resource/df7.json')
df11_cfg = load_force_config('./resource/df11.json')
# domain_4 = load_domain_config('./resource/domain_4.json')
# domain_mrcg = load_domain_config('./resource/domain_mrcg.json')
domain_alt = load_domain_config('./resource/domain_alt.json')
# domain_mrcg_gw = load_domain_config('./resource/domain_mrcg_gw.json')
# domain_basic = load_domain_config('./resource/domain_basic.json')

def evolve_domain(domain_cfg: DomainConfig, force_cfg: ForceConfig):
    preprocess(domain_cfg, force_cfg)
    # warmup_solver(domain_cfg, force_cfg)
    solver(domain_cfg, force_cfg)

if __name__ == '__main__':
    # evolve_domain(domain_alt, df7_cfg)
    
    # draw_hydrograph(domain_alt, 'D74', True, -3600)
    # generate_flood_map(domain_alt)
    # generate_flood_video(domain_alt, output_path='./resource/flood_video.mp4')
    
    # preprocess(domain_mrcg, df7_cfg)
    # generate_max_inundation_extent_map(domain_4)
    
    mses = compare_hydrograph([domain_alt], 'D74', clampped=True, show=True, show_obs=True)
    print(f'RMSEs: {mses}')
    
    # plot_spatial_mae_curve(domain_4, domain_mrcg, df7_cfg, output_path='./resource/spatial_mae_curve.png')
    