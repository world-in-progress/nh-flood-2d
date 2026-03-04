from src.nh_flood_2d.core.solver_compact import solver
from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.input import load_input_config
from src.nh_flood_2d.output.flood_map import generate_flood_map
from src.nh_flood_2d.output.hydrograph import draw_hydrograph

if __name__ == '__main__':
    config = load_input_config('./resource/config.json')
    
    # preprocess(config)
    solver(config)
    # # generate_flood_map(config)
    
    stations = ['R22', 'D73', 'D74', 'D82']
    draw_hydrograph(config, stations[0], True)