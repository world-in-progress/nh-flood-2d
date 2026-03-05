from src.nh_flood_2d.core.solver_compact import solver
from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.input import load_input_config, DomainConfig
from src.nh_flood_2d.output.flood_map import generate_flood_map
from src.nh_flood_2d.output.hydrograph import draw_hydrograph, compare_hydrograph

domain_4 = load_input_config('./resource/domain_4.json')
domain_mrgc = load_input_config('./resource/domain_mrgc.json')

def evolve_domain(cfg: DomainConfig):
    preprocess(cfg)
    solver(cfg)
    generate_flood_map(cfg)

if __name__ == '__main__':
    mses = compare_hydrograph([domain_4, domain_mrgc], 'R22', forward_ignore_second=3600 * 4)
    print(f'RMSEs: {mses}')