import math
import os

import numpy as np
from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
import geopy.distance
from tqdm import tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor
import threading
from typing import List
import requests

from osc_downloader import Coordinates, OSCDownloadManager

THREAD_LOCK = threading.Lock()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    split = []
    for i in range(0, len(lst), n):
        split.append(lst[i:i + n])
    return split

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def Haversine(coords1, coords2):
    """
    This uses the ‘haversine’ formula to calculate the great-circle distance between two points – that is,
    the shortest distance over the earth’s surface – giving an ‘as-the-crow-flies’ distance between the points
    (ignoring any hills they fly over, of course!).
    Haversine
    formula:    a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    c = 2 ⋅ atan2( √a, √(1−a) )
    d = R ⋅ c
    where   φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km);
    note that angles need to be in radians to pass to trig functions!
    """
    R = 6371.0088
    dcoords = coords2 - coords1
    lat1, lon1 = coords1[:, 0], coords1[:, 1]
    lat2, lon2 = coords2[:, 0], coords2[:, 1]

    dlat = dcoords[:, 0]
    dlon = dcoords[:, 1]
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) **2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    d = (R * c) * 1000
    return np.round(d,4)


def elements_to_numpy(elements):
    arr = np.array([
        [c["lat"], c["lon"]] for elem in elements for c in elem["geometry"]
    ])
    return arr

def get_elem_idxs(elements):
    out = [[i]*len(elem["geometry"]) for i, elem in enumerate(
        elements)]
    out = [e for o in out for e in o]
    return out


def find_closest_element(elements: List, coord: Coordinates):
    elem_coords = elements_to_numpy(elements)
    coord = np.array([[coord.lat, coord.long]] * elem_coords.shape[0])

    d = haversine_np(lon1=elem_coords[:, 1], lat1=elem_coords[:, 0],
                     lon2=coord[:, 1], lat2=coord[:, 0])

    idx = np.argmin(d)
    elem_idxs = get_elem_idxs(elements)

    closest_elem = elements[elem_idxs[idx]]
    closest_dist = d[idx]
    return closest_dist, closest_elem


class Bbox:
    def __init__(self, min_lat: float, min_long: float, max_lat: float,
                 max_long: float):
        self.min_lat = min_lat
        self.min_long = min_long
        self.max_lat = max_lat
        self.max_long = max_long

    def __getitem__(self, item):
        if item < 0:
            item = 4 - item

        if item < 0 or item > 3:
            raise IndexError

        if item == 0:
            return self.min_lat
        elif item == 1:
            return self.min_long
        elif item == 2:
            return self.max_lat
        elif item == 3:
            return self.max_long

    def __repr__(self):
        return f"[{self.min_lat}, {self.min_long}, {self.max_lat}, {self.max_long}]"


class OSMDownloadManager(threading.Thread):
    def __init__(self, coordinates: List[Coordinates], max_workers: int = 10):
        super(OSMDownloadManager, self).__init__()
        self.coordinates = coordinates
        self.max_workers = max_workers
        self.download_complete = threading.Event()

    def start_download(self):
        # create bboxes based on coordinates
        bboxes = [
            Bbox(min_lat=c.lat, min_long=c.long,
                 max_lat=c.lat+0.000001, max_long=c.long-0.000001)
            for c in self.coordinates
            ]

        # store the mapping of coordinates to bboxes for later usage
        coord_bbox_mappings = {c: b for c, b in zip(self.coordinates, bboxes)}

        # query the metadata for the bboxes
        pbar = tqdm(total=len(bboxes), desc="Querying OSM metadata")
        query_handler = OSMQueryHandler(bboxes=bboxes, max_workers=self.max_workers,
                                        pbar=pbar)
        bboxes_responses = query_handler.query()
        pbar.close()

        # store the mapping of bboxes to osm responses for later usage
        bbox_response_mappings = {item["bbox"]: item["response"]
                                  for item in bboxes_responses}

        # of al retrieved elements, find the way with the closest match to our
        # coordinate
        coord_way_mappings = {}
        for coord in tqdm(self.coordinates, desc="Determining OSM elements for "
                                                 "coordinates"):
            osm_response = bbox_response_mappings[coord_bbox_mappings[coord]]
            elem = None
            if osm_response:
                elements = osm_response["elements"]
                dist, elem = find_closest_element(elements, coord)
            coord_way_mappings[coord] = elem

        self.coord_way_mappings = coord_way_mappings

        return coord_way_mappings


    def _make_splits(self, bboxes):
        total = len(bboxes)
        split_size = math.ceil(total / self.max_workers)
        return chunks(bboxes, split_size)

    def run(self) -> None:
        self.start_download()
        self.download_complete.set()


class OSMQueryHandler:
    def __init__(self, bboxes: List, max_workers: int = 10, pbar: tqdm = None):
        self.bboxes = bboxes
        self.max_workers = max_workers
        self.pbar = pbar

    def _make_splits(self, bboxes):
        total = len(bboxes)
        split_size = math.ceil(total / self.max_workers)
        return chunks(bboxes, split_size)

    def query(self):
        bbox_splits = self._make_splits(self.bboxes)
        query_operation = OSMQueryOperation(pbar=self.pbar)


        query_operation = OSMQueryOperation(pbar=self.pbar)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executors:
            futures = [executors.submit(query_operation.query_by_bbox, bboxes)
                       for bboxes in bbox_splits]
            bboxes_responses = []
            for future in as_completed(futures):
                bboxes_responses += future.result()

        return bboxes_responses


class OSMQueryOperation:
    def __init__(self, pbar: tqdm = None, timeout: int = 300):
        self.pbar = pbar
        self.timeout = timeout
        self.overpass = Overpass()

    def query_by_bbox(self, bboxes: List, elem_type: str = "way",
                      conditions: str = "count_tags() > 6",
                      include_geom: bool = True):
        responses = []
        for bbox in bboxes:
            query = overpassQueryBuilder(
                bbox=bbox, elementType=elem_type, conditions=conditions,
                out='body', includeGeometry=include_geom
            )
            response = self._single_query(query)
            if response:
                response = response.toJSON()

            responses.append({"bbox": bbox, "response": response})

            if self.pbar:
                with THREAD_LOCK:
                    self.pbar.update(1)

        return responses

    def _single_query(self, query: str):
        try:
            response = self.overpass.query(query, timeout=self.timeout)
            return response
        except:
            return False


if __name__ == "__main__":

    bbox = [35.224792, -114.275691, 35.225285, -114.281114]
    bbox = [35.224792, -114.275691, 35.224793, -114.275692]

    coordinates = [Coordinates(35.224792, -114.275691)]
    osm_download_manager = OSMDownloadManager(coordinates)
    osm_download_manager.start_download()

    # osc_download_manager = OSCDownloadManager(
    #     coordinates=coordinates, max_workers=max_workers,
    #     photo_dir=os.path.join(out_dir, "photos")
    # )