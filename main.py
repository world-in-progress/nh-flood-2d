from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.core.solver_compact import solver
from src.nh_flood_2d.output.flood_map import generate_flood_map
from src.nh_flood_2d.output.hydrograph import draw_hydrograph, compare_hydrograph
from src.nh_flood_2d.input import load_domain_config, DomainConfig, load_force_config, ForceConfig

df7_cfg = load_force_config('./resource/df7.json')
df11_cfg = load_force_config('./resource/df11.json')
domain_4 = load_domain_config('./resource/domain_4.json')
domain_mrcg = load_domain_config('./resource/domain_mrcg.json')
domain_basic = load_domain_config('./resource/domain_basic.json')

def evolve_domain(domain_cfg: DomainConfig, force_cfg: ForceConfig):
    preprocess(domain_cfg, force_cfg)
    solver(domain_cfg, force_cfg)

if __name__ == '__main__':
    # evolve_domain(domain_mrcg, df7_cfg)
    
    # draw_hydrograph(domain_mrcg, 'D74', True, -3600)
    # generate_flood_map(domain_mrcg)
    
    mses = compare_hydrograph([domain_mrcg], 'R22', clampped=True, show=True)
    print(f'RMSEs: {mses}')
    