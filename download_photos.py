from osc_downloader import OSCDownloadManager
from osc_downloader import Coordinates

import os
import argparse


def get_coordinates_from_txt(path):
    coords = []
    with open(path, "r") as f:
        for line in f:
            c = line.replace("\n", "")
            c = c.split(",")
            coords.append(Coordinates(lat=float(c[0]), long=float(c[1])))

    return coords


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--coordinates", type=str,
                        help="Path to .txt file containing the coordinates.",
                        default="test_coordinates.txt")
    parser.add_argument("-o", "--out-dir", type=str,
                        help="Directory to save the downloaded stuff to.",
                        default="downloads")
    opts = parser.parse_args()

    coords = get_coordinates_from_txt(opts.coordinates)

    manager = OSCDownloadManager(coordinates=coords, search_radius=10,
                                 photo_dir=opts.out_dir)

    manager.start_download()
