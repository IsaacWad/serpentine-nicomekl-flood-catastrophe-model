"""Run the cleaned Serpentine-Nicomekl conditional flood-event model."""
import argparse
import csv
import gc
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ['raw_segments_clipped_to_flood_zone_clean_gdf.geojson', 'serpentine_nicomekl_hazard.geojson',
          'coastal_storage_side_mask.gpkg', 'base_exposure_with_hazus.geojson',
          'hazus_full_depth_damage_curves.csv', 'serpentine_nicomekl_merged_dem_epsg3005.tif',
          'dike_gdf.geojson', 'appurtenant_gdf.geojson']


def check_files(data_dir):
    missing = [str(data_dir / name) for name in INPUTS if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing inputs: ' + ', '.join(missing))
    schemas = {
        'raw_segments_clipped_to_flood_zone_clean_gdf.geojson': ['segment_id', 'event_source'],
        'base_exposure_with_hazus.geojson': ['exposure_id', 'building_tiv', 'hazus_occupancy', 'hazus_a_zone_building_dmg_fn_id'],
        'serpentine_nicomekl_hazard.geojson': [], 'dike_gdf.geojson': [], 'appurtenant_gdf.geojson': []}
    counts = {}
    for name, fields in schemas.items():
        data = json.loads((data_dir / name).read_text(encoding='utf-8'))
        if not data.get('features'):
            raise ValueError(f'{name}: empty feature collection')
        for feature in data['features']:
            if not feature.get('geometry') or set(fields) - set(feature['properties']):
                raise ValueError(f'{name}: missing geometry or required attributes')
        counts[name] = len(data['features'])
    with (data_dir / 'hazus_full_depth_damage_curves.csv').open(encoding='utf-8-sig', newline='') as handle:
        columns = next(csv.reader(handle))
        if not {'Occupancy', 'DmgFnId', 'ft00'}.issubset(columns):
            raise ValueError('HAZUS curve columns are incomplete')
    return counts


def load_inputs(data_dir):
    import geopandas as gpd
    import pandas as pd
    import rasterio
    layers = {name:gpd.read_file(data_dir / name).to_crs('EPSG:3005') for name in
              ['raw_segments_clipped_to_flood_zone_clean_gdf.geojson', 'serpentine_nicomekl_hazard.geojson',
               'coastal_storage_side_mask.gpkg', 'base_exposure_with_hazus.geojson', 'dike_gdf.geojson', 'appurtenant_gdf.geojson']}
    with rasterio.open(data_dir / 'serpentine_nicomekl_merged_dem_epsg3005.tif') as raster:
        if raster.crs is None or raster.crs.to_epsg() != 3005:
            raise ValueError('DEM must use EPSG:3005; reprojection of vector layers does not reproject the raster')
    curves = pd.read_csv(data_dir / 'hazus_full_depth_damage_curves.csv')
    return layers, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed/simulation')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--n-sims', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check-files', action='store_true', help='Check file presence and GeoJSON schemas using the standard library')
    parser.add_argument('--check-inputs', action='store_true', help='Also load geospatial layers and check DEM CRS; no scoring or simulation')
    parser.add_argument('--save-affected', action='store_true', help='Save potentially large affected-property files for each batch')
    args = parser.parse_args()
    if args.n_sims < 1 or args.batch_size < 1:
        parser.error('simulation count and batch size must be positive')
    counts = check_files(args.data_dir)
    if args.check_files:
        print(json.dumps(counts, indent=2))
        return
    layers, curves = load_inputs(args.data_dir)
    if args.check_inputs:
        import final_monte_carlo_simulation
        import assign_segment_score
        print('Geospatial inputs and model imports validated:', counts)
        return
    import geopandas as gpd
    import pandas as pd
    import assign_segment_score
    import final_monte_carlo_simulation as mc
    from summarize_results import summarize

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dem = args.data_dir / 'serpentine_nicomekl_merged_dem_epsg3005.tif'
    segments = assign_segment_score.assign_segment_scores(
        segments_gdf=layers['raw_segments_clipped_to_flood_zone_clean_gdf.geojson'], dem_path=dem,
        dike_gdf=layers['dike_gdf.geojson'], appurtenant_gdf=layers['appurtenant_gdf.geojson'],
        low_elevation_threshold_m=2.0, sample_spacing_m=50, protection_buffer_distance_m=500, crs='EPSG:3005')
    segments.to_file(output / 'scored_clean_segments.geojson', driver='GeoJSON')
    boundary = layers['serpentine_nicomekl_hazard.geojson'].geometry.union_all().buffer(5000)
    config = {'n_sims':args.n_sims, 'batch_size':args.batch_size, 'base_seed':args.seed,
              'batch_seed_rule':'base_seed + batch_index', 'expansion_boundary_buffer_m':5000,
              'segments':'cleaned raw source segments, rescored at runtime',
              'event_source_weights':mc.select_segment.DEFAULT_EVENT_SOURCE_WEIGHTS,
              'severity_parameters':mc.DEFAULT_SEVERITY_PARAMETERS,
              'footprint_parameters':mc.DEFAULT_FOOTPRINT_PARAMETERS,
              'interpretation':'conditional event gross building loss, not annual insured loss'}
    (output / 'run_configuration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    batches = []
    for batch_index, offset in enumerate(range(0, args.n_sims, args.batch_size)):
        size = min(args.batch_size, args.n_sims - offset)
        events, affected, footprints, damage = mc.run_monte_carlo_simulation(
            n_sims=size, scored_segments_gdf=segments, exposure_gdf=layers['base_exposure_with_hazus.geojson'],
            hazus_curves_df=curves, dem_path=dem, expansion_boundary=boundary,
            coastal_storage_side_gdf=layers['coastal_storage_side_mask.gpkg'], tiv_col='building_tiv',
            random_seed=args.seed + batch_index, print_progress=True, continue_on_error=True)
        events['batch_id'] = batch_index
        events['sim_id_global'] = events['sim_id'] + offset
        events.to_csv(output / f'event_results_batch_{batch_index:03d}.csv', index=False)
        batches.append(events)
        if args.save_affected and not affected.empty:
            affected['batch_id'] = batch_index
            affected['sim_id_global'] = affected['sim_id'] + offset
            affected.to_file(output / f'affected_properties_batch_{batch_index:03d}.geojson', driver='GeoJSON')
        del affected, footprints, damage
        gc.collect()
    master = pd.concat(batches, ignore_index=True)
    path = output / 'event_results_master.csv'
    master.to_csv(path, index=False)
    summary = summarize(path, output)
    print(json.dumps(summary, indent=2))
    if summary['succeeded'] == 0:
        raise RuntimeError('Every simulated event failed; inspect event_results_master.csv')


if __name__ == '__main__':
    main()
