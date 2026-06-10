import numpy as np
import pandas as pd
import taichi as ti
import fastdb4py as fdb
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

from ..input import DomainConfig
from ..util.ti import init_taichi, copy_to_taichi
from ..schema.feature import UVH, Ne, Ns, IndexLike

def _find_ei(ne_fdb_path: str, ns_fdb_path: str, x: float, y: float) -> int:
    init_taichi(use_gpu=True)
    
    ne_fdb = fdb.ORM.load(ne_fdb_path, from_file=True)
    ns_fdb = fdb.ORM.load(ns_fdb_path, from_file=True)
    
    nes = ne_fdb[Ne][Ne]
    nss = ns_fdb[Ns][Ns]
    
    e_num = len(nes)
    
    exs = copy_to_taichi(nes.column.x, ti.f32, None)
    eys = copy_to_taichi(nes.column.y, ti.f32, None)
    sxs = copy_to_taichi(nss.column.x, ti.f32, None)
    sys = copy_to_taichi(nss.column.y, ti.f32, None)
    isl_data  = copy_to_taichi(ne_fdb[IndexLike]['isl_data'].column.index,  ti.i32, None)
    isl_ptr_l = copy_to_taichi(ne_fdb[IndexLike]['isl_ptr_l'].column.index, ti.i32, None)
    isl_ptr_b = copy_to_taichi(ne_fdb[IndexLike]['isl_ptr_b'].column.index, ti.i32, None)

    the_ei = ti.field(ti.i32, shape=())
    the_ei[None] = -1

    @ti.kernel
    def get_ei():
        for ei in range(1, e_num):
            lsi0 = isl_data[isl_ptr_l[ei]]   # first left side
            lsi2 = isl_data[isl_ptr_b[ei]]   # first bottom side
            slh = ti.floor(exs[ei] - sxs[lsi0] + 0.5) * 2.0
            slv = ti.floor(eys[ei] - sys[lsi2] + 0.5) * 2.0
            
            xmin = exs[ei] - slh * 0.5
            ymin = eys[ei] - slv * 0.5
            xmax = exs[ei] + slh * 0.5
            ymax = eys[ei] + slv * 0.5
            
            if xmin <= x and x <= xmax and ymin <= y and y <= ymax:
                the_ei[None] = ei
    
    get_ei()
    return the_ei[None]

def _extract_data(cfg: DomainConfig, station_name: str):
    db_path = Path(cfg.uvh_dir)
    
    # Find the hydro element index (the_ei) containing the hydrograph point
    px, py = cfg.hydrograph_points[station_name]
    the_ei = _find_ei(cfg.ne_fdb, cfg.ns_fdb, px, py)
    if the_ei == -1:
        raise ValueError(f'No hydro element found containing point ({px}, {py}) for station {station_name}')
    
    # Extract and sort times from uvh fdb file names
    times: list[str] = []
    for db_file in db_path.glob('*.fdb'):
        time_str = db_file.stem.split('_')[-1]
        times.append(time_str)
    times.sort(
        key=lambda x: datetime.strptime(x, '%Y%m%d-%H%M%S').timestamp()
    )
    
    # Extract water depth at the_ei for each timestamp and save to txt file
    hs: list[float] = []
    for time_str in times:
        db_file = str(db_path / f'uvh_{time_str}.fdb')
        db = fdb.ORM.load(db_file, from_file=True)
        
        uvh = db[UVH][UVH].column.h
        h = uvh[the_ei]
        hs.append(float(h))
    
    with open(f'{cfg.hydrograph_dir}/{station_name}.txt', 'w') as f:
        for time_str, h in zip(times, hs):
            f.write(f'{time_str}, {h}\n')

def _load_observation_water_depth(observation_file: Path) -> pd.DataFrame:
    df = pd.read_csv(observation_file)
    if df.empty:
        return pd.DataFrame(columns=['datetime', 'Waterdepth'])

    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        dayfirst=True
    )

    water_depth_column = next(
        (
            column_name
            for column_name in df.columns
            if 'water' in column_name.lower()
            and ('depth' in column_name.lower() or 'level' in column_name.lower())
        ),
        None,
    )
    if water_depth_column is None:
        raise ValueError(
            f'No water depth column found in observation file {observation_file}'
        )

    df['Waterdepth'] = pd.to_numeric(df[water_depth_column], errors='coerce')
    obs_df = df[['datetime', 'Waterdepth']].dropna(subset=['Waterdepth']).copy()
    obs_df.sort_values('datetime', inplace=True)
    return obs_df
            
def draw_hydrograph(cfg: DomainConfig, station_name: str, clampped: bool = True, translation_second: int = 0):
    observation_file = Path(cfg.observation_dir) / f'{station_name}.csv'
    if not observation_file.exists():
        raise FileNotFoundError(f'Observation file for station {station_name} not found at {observation_file}')
    
    # Load observed water depth data.
    obs_df = _load_observation_water_depth(observation_file)
    
    # Extract and load simulation data
    _extract_data(cfg, station_name)
    sim_df = pd.read_csv(f'{cfg.hydrograph_dir}/{station_name}.txt', header=None, names=['datetime', 'depth'])
    sim_df['datetime'] = pd.to_datetime(sim_df['datetime'], format='%Y%m%d-%H%M%S') + pd.to_timedelta(translation_second, unit='s')
    sim_df.sort_values('datetime', inplace=True)
    
    # Clamp time range to the overlapping period of observation and simulation data if clampped is True
    if clampped:
        start_time = max(
            obs_df['datetime'].min(),
            sim_df['datetime'].min()
        )
        end_time = min(
            obs_df['datetime'].max(),
            sim_df['datetime'].max()
        )
        print(f'Clamping time range to {start_time} - {end_time}')
        obs_df = obs_df.query('@start_time <= datetime <= @end_time')
        sim_df = sim_df.query('@start_time <= datetime <= @end_time')
    
    plt.figure(figsize=(12, 6))
    plt.plot(
        obs_df['datetime'], obs_df['Waterdepth'],
        label='Observed Water Depth (m)', linewidth=2
    )
    plt.plot(
        sim_df['datetime'], sim_df['depth'],
        label='Simulated Water Depth (m)', linewidth=2
    )
    
    plt.title(f'Hydrograph at Station {station_name}', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Water Depth (m)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

def _prepare_hydrograph_comparison_data(
    cfgs: list[DomainConfig],
    station_name: str,
    clampped: bool = True,
    translation_second: int = 0,
    forward_ignore_second: int = 0,
    show_obs: bool = True,
) -> tuple[pd.DataFrame | None, list[tuple[str, pd.DataFrame]]]:
    if not cfgs:
        raise ValueError('At least one domain config is required for hydrograph comparison')

    # Validate that all configs reference the same observation file
    existing_obs_paths = [
        str(Path(cfg.observation_dir) / f'{station_name}.csv')
        for cfg in cfgs
        if (Path(cfg.observation_dir) / f'{station_name}.csv').exists()
    ]
    unique_obs_paths = list(dict.fromkeys(existing_obs_paths))
    if len(unique_obs_paths) > 1:
        raise ValueError(
            f'Configs reference different observation files for station {station_name}: {unique_obs_paths}. '
            'All configs must use the same observation data.'
        )

    # Load observed water depth data from the single validated observation file.
    obs_df = None
    if show_obs and unique_obs_paths:
        observation_file = Path(unique_obs_paths[0])
        obs_df = _load_observation_water_depth(observation_file)

    # Extract and load simulation data for each config
    sim_dfs: list[tuple[str, pd.DataFrame]] = []
    for cfg in cfgs:
        _extract_data(cfg, station_name)
        sim_df = pd.read_csv(f'{cfg.hydrograph_dir}/{station_name}.txt', header=None, names=['datetime', 'depth'])
        sim_df['datetime'] = pd.to_datetime(sim_df['datetime'], format='%Y%m%d-%H%M%S') + pd.to_timedelta(translation_second, unit='s')
        sim_df.sort_values('datetime', inplace=True)

        # Drop the first forward_ignore_second seconds of simulation data
        if forward_ignore_second > 0:
            ignore_cutoff = sim_df['datetime'].min() + pd.to_timedelta(forward_ignore_second, unit='s')
            sim_df = sim_df[sim_df['datetime'] >= ignore_cutoff]

        label = Path(cfg.domain_dir).name
        sim_dfs.append((label, sim_df))

    # Clamp time range to the overlapping period of all data if clampped is True
    if clampped:
        all_mins = [s['datetime'].min() for _, s in sim_dfs if not s.empty]
        all_maxs = [s['datetime'].max() for _, s in sim_dfs if not s.empty]
        if obs_df is not None and not obs_df.empty:
            all_mins.append(obs_df['datetime'].min())
            all_maxs.append(obs_df['datetime'].max())
        if all_mins and all_maxs:
            start_time = max(all_mins)
            end_time = min(all_maxs)
            print(f'Clamping time range to {start_time} - {end_time}')
            if obs_df is not None:
                obs_df = obs_df[(obs_df['datetime'] >= start_time) & (obs_df['datetime'] <= end_time)]
            sim_dfs = [
                (label, s[(s['datetime'] >= start_time) & (s['datetime'] <= end_time)])
                for label, s in sim_dfs
            ]

    return obs_df, sim_dfs

def compare_hydrograph(
    cfgs: list[DomainConfig], station_name: str,
    clampped: bool = True, 
    translation_second: int = 0, forward_ignore_second: int = 0,
    show_obs: bool = True,
    baseline: DomainConfig | None = None,
    show: bool = True
) -> tuple[list[float], list[float]]:
    obs_df, sim_dfs = _prepare_hydrograph_comparison_data(
        cfgs,
        station_name,
        clampped=clampped,
        translation_second=translation_second,
        forward_ignore_second=forward_ignore_second,
        show_obs=show_obs,
    )

    plt.figure(figsize=(12, 6))
    if obs_df is not None:
        plt.plot(
            obs_df['datetime'], obs_df['Waterdepth'],
            label='Observed Water Depth (m)', linewidth=2
        )
    for label, sim_df in sim_dfs:
        plt.plot(
            sim_df['datetime'], sim_df['depth'],
            label=f'Simulated Water Depth (m) [{label}]', linewidth=2
        )

    if show:
        ax = plt.gca()
        ax.set_title(
            f'Water Depth Hydrograph Comparison at {station_name}',
            fontsize=16,
            fontname='Times New Roman',
            fontweight='bold'
        )
        ax.set_xlabel('Time', fontsize=14, fontname='Times New Roman', fontweight='bold')
        ax.set_ylabel('Water Depth (m)', fontsize=14, fontname='Times New Roman', fontweight='bold')
        ax.legend(prop={'family': 'Times New Roman', 'weight': 'bold'})
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontname('Times New Roman')
            tick_label.set_fontweight('bold')
        plt.tight_layout()
        plt.show()
    
    # Compute RMSE and NSE values against the baseline series using linear interpolation on time
    def _nse(obs: np.ndarray, sim: np.ndarray) -> float:
        denom = np.sum((obs - obs.mean()) ** 2)
        return float(1.0 - np.sum((obs - sim) ** 2) / denom) if denom != 0 else float('-inf')

    if baseline is not None:
        # Use the specified baseline config's simulation data as reference
        baseline_label = Path(baseline.domain_dir).name
        baseline_entry = next(
            ((lbl, df) for lbl, df in sim_dfs if lbl == baseline_label), None
        )
        if baseline_entry is None:
            raise ValueError(
                f'Baseline config "{baseline_label}" was not found in cfgs. '
                'The baseline must be one of the provided configs.'
            )
        _, baseline_df = baseline_entry

        base_ts = baseline_df['datetime'].astype(np.int64).values / 1e9
        base_vals = baseline_df['depth'].values

        rmse_values: list[float] = []
        nse_values: list[float] = []
        for lbl, sim_df in sim_dfs:
            if lbl == baseline_label:
                continue
            sim_ts = sim_df['datetime'].astype(np.int64).values / 1e9
            interp_vals = np.interp(base_ts, sim_ts, sim_df['depth'].values)
            rmse_values.append(float(np.sqrt(np.mean((base_vals - interp_vals) ** 2))))
            nse_values.append(_nse(base_vals, interp_vals))
        return rmse_values, nse_values
    else:
        # Use obs as baseline; return [] if obs is unavailable
        if obs_df is None:
            return [], []

        base_ts = obs_df['datetime'].astype(np.int64).values / 1e9
        base_vals = obs_df['Waterdepth'].values

        rmse_values = []
        nse_values = []
        for lbl, sim_df in sim_dfs:
            sim_ts = sim_df['datetime'].astype(np.int64).values / 1e9
            interp_vals = np.interp(base_ts, sim_ts, sim_df['depth'].values)
            rmse_values.append(float(np.sqrt(np.mean((base_vals - interp_vals) ** 2))))
            nse_values.append(_nse(base_vals, interp_vals))
        return rmse_values, nse_values

def compare_hydrograph_panels(
    cfgs: list[DomainConfig],
    station_names: list[str],
    clampped: bool = True,
    translation_second: int = 0,
    forward_ignore_second: int = 0,
    show_obs: bool = False,
    show: bool = True,
    output_path: str | None = None,
    title: str = 'Water Depth Hydrograph Comparison',
    figsize: tuple[float, float] | None = None,
    dpi: int = 300,
) -> Figure:
    if not station_names:
        raise ValueError('At least one station name is required for panel plotting')

    station_plots: list[tuple[str, pd.DataFrame | None, list[tuple[str, pd.DataFrame]]]] = []
    all_depth_values: list[float] = []
    all_start_times: list[pd.Timestamp] = []
    all_end_times: list[pd.Timestamp] = []

    for station_name in station_names:
        obs_df, sim_dfs = _prepare_hydrograph_comparison_data(
            cfgs,
            station_name,
            clampped=clampped,
            translation_second=translation_second,
            forward_ignore_second=forward_ignore_second,
            show_obs=show_obs,
        )
        station_plots.append((station_name, obs_df, sim_dfs))

        if obs_df is not None and not obs_df.empty:
            all_depth_values.extend(obs_df['Waterdepth'].tolist())
            all_start_times.append(obs_df['datetime'].min())
            all_end_times.append(obs_df['datetime'].max())
        for _, sim_df in sim_dfs:
            if sim_df.empty:
                continue
            all_depth_values.extend(sim_df['depth'].tolist())
            all_start_times.append(sim_df['datetime'].min())
            all_end_times.append(sim_df['datetime'].max())

    if not all_depth_values:
        raise ValueError('No hydrograph data available for panel plotting')

    if figsize is None:
        figsize = (10 * len(station_names), 5.4)

    depth_min = min(all_depth_values)
    depth_max = max(all_depth_values)
    depth_padding = max((depth_max - depth_min) * 0.05, 0.05)
    start_time = min(all_start_times)
    end_time = max(all_end_times)

    fig, axes = plt.subplots(
        1,
        len(station_names),
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor('white')

    for axis, (station_name, obs_df, sim_dfs) in zip(axes, station_plots):
        if obs_df is not None:
            axis.plot(
                obs_df['datetime'],
                obs_df['Waterdepth'],
                label='Observed Water Depth (m)',
                linewidth=2,
            )
        for label, sim_df in sim_dfs:
            axis.plot(
                sim_df['datetime'],
                sim_df['depth'],
                label=f'Simulated Water Depth (m) [{label}]',
                linewidth=2,
            )

        axis.set_xlim(start_time, end_time)
        axis.set_ylim(depth_min - depth_padding, depth_max + depth_padding)
        axis.set_facecolor('white')
        axis.set_xlabel('Time', fontsize=13, fontname='Times New Roman', fontweight='semibold')
        axis.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.28)
        axis.tick_params(axis='both', which='major', labelsize=11, width=1.0)
        axis.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
        axis.text(
            0.98,
            0.02,
            station_name,
            transform=axis.transAxes,
            ha='right',
            va='bottom',
            fontsize=17,
            fontname='Times New Roman',
            fontweight='bold',
        )
        for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
            tick_label.set_fontname('Times New Roman')
            tick_label.set_fontweight('normal')

    axes[0].set_ylabel('Water Depth (m)', fontsize=13, fontname='Times New Roman', fontweight='semibold')

    # Use a single, external legend to avoid covering hydrograph lines.
    legend_handles: list = []
    legend_labels: list[str] = []
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.91),
            ncol=max(1, min(3, len(legend_labels))),
            frameon=True,
            prop={'family': 'Times New Roman', 'weight': 'semibold', 'size': 11},
        )

    fig.suptitle(
        title,
        fontsize=20,
        fontname='Times New Roman',
        fontweight='semibold',
        y=0.965,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()

    return fig