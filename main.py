from src.nh_flood_2d.output.flood_map import generate_flood_map

if __name__ == '__main__':
    generate_flood_map(
        ne_fdb_fn='./resource/fdb/ne.fdb',
        ns_fdb_fn='./resource/fdb/ns.fdb',
        uvhs_dir='./resource/uvh/',
        epsg_code=2326,
        output_dir='./resource/output/'
    )