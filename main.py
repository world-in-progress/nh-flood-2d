from src.nh_flood_2d.core import solver
from src.nh_flood_2d.core.domain import Domain
from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.input import load_input_config
from src.nh_flood_2d.output.flood_map import generate_flood_map
from src.nh_flood_2d.output.hydrograph import draw_hydrograph

if __name__ == '__main__':
    config = load_input_config('./resource/config.json')
    
    
    # domain = Domain(config)
    # for t in domain.evolve():
    #     print(f'Time step {t} completed.')
    
    # preprocess(config)
    # solver(config)    
    generate_flood_map(config)
    # draw_hydrograph(config, 'D73', True, -3600)