"""Run the configured two-dimensional model."""
import argparse

from src.nh_flood_2d.input import load_domain_config, load_force_config
from src.nh_flood_2d.core.solver_compact import run_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', required=True, help='Domain JSON configuration')
    parser.add_argument('--force', required=True, help='Boundary/rainfall JSON configuration')
    parser.add_argument('--pipe', help='Optional SWMM pipe configuration')
    args = parser.parse_args()
    domain, force = load_domain_config(args.domain), load_force_config(args.force)
    from src.nh_flood_2d.input.pipe import load_pipe_config
    pipe = load_pipe_config(args.pipe) if args.pipe else None
    run_simulation(domain, force, pipe)


if __name__ == '__main__':
    main()
