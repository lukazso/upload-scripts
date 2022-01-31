import argparse
import os.path

import pandas

from osc_downloader import OSCDownloadManager
from osm_downloader import OSMDownloadManager
from osc_downloader import Coordinates

DESC = "This script helps you to download all photos (open street cam) and " \
       "road attributes (open street maps) given a list of coordinates."



def coordinates_from_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError

    coords = []
    with open(path, "r") as f:
        for line in f:
            line = line.replace("\n", "").split()
            if len(line) != 2:
                raise SyntaxError("Incorrect syntax in coordinates file.")

            lat = float(line[0])
            long = float(line[1])
            coords.append(Coordinates(lat=lat, long=long))

    return coords


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--coordinate-file", type=str,
                        default="test_coordinates.txt",
                        help=".txt file where coordinates are stored.")
    parser.add_argument("-o", "--out-dir", type=str,
                        default="downloads",
                        help="Directory to store the results in.")
    parser.add_argument("-w", "--worker", type=int,
                        default=10,
                        help="Number of threads to use.")

    opts = parser.parse_args()

    coords = coordinates_from_file(path=opts.coordinate_file)

    osc_manager = OSCDownloadManager(
        coordinates=coords, max_workers=opts.worker,
        photo_dir=os.path.join(opts.out_dir, "photos"), show_pbar=False
    )
    osm_manager = OSMDownloadManager(
        coordinates=coords, max_workers=min(2, opts.worker)
    )

    print("Starting photo download...")
    osc_manager.start()
    print("Starting metadata download...")
    osm_manager.start()

    osc_manager.join()
    osm_manager.join()

    print("Creating database...")

    # get all available tags
    tags = list(set([e for elem in osm_manager.coord_way_mappings.values()
                     for e in elem["tags"].keys()]))

    columns = ["coordinates", "photos"] + tags

    rows = []
    empty_row = {k: None for k in columns}
    for coord, photo_paths in osc_manager.coords_path_mapping.items():
        new_row = empty_row.copy()
        new_row["coordinates"] = coord
        new_row["photos"] = photo_paths

        elem = osm_manager.coord_way_mappings[coord]
        for tag, value in elem["tags"].items():
            new_row[tag] = value

        rows.append(new_row)

    df = pandas.DataFrame.from_dict(rows, orient="columns")

    out_path = os.path.join(opts.out_dir, "database.csv")
    df.to_csv(out_path)
    print(f"Saved database to {out_path}.")
