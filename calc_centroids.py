import geopandas as gpd
import argparse
import os


def get_centroid(x):
    return x.centroid

def main(gdf):
    gdf['centroid'] = gdf.geometry.apply(lambda x: get_centroid(x))
    gdf['c_long'] = gdf['centroid'].astype(str).str.split("(").str[1].str.strip(")").str.split(" ").str[0]
    gdf['c_lat'] = gdf['centroid'].astype(str).str.split("(").str[1].str.strip(")").str.split(" ").str[1]
    gdf = gdf.drop(["centroid"], axis = 1)
    gdf["boxID"] = [i for i in range(0, len(gdf))]
    return gdf


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("iso", help="Country ISO")
    args = parser.parse_args()

    shp_path = os.path.join("./data/", args.iso, (args.iso + "imagery_bboxes.shp"))
    gdf = gpd.read_file(shp_path)

    new_gdf = main(gdf)
       
    new_path = os.path.join("./data/", args.iso, (args.iso + "imagery_bboxes.shp"))
    new_gdf.to_file(new_path)