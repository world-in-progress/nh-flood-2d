from src.nh_flood_2d.core import solver
from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.input import load_input_config
from src.nh_flood_2d.output.flood_map import generate_flood_map

if __name__ == '__main__':
    config = load_input_config('./resource/config.json')
    
    preprocess(config)
    solver(config)
    # generate_flood_map(config)