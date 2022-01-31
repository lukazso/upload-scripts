import logging
import json
import math
import os
import threading
from concurrent.futures import as_completed, ThreadPoolExecutor
# third party
from typing import List
from OSMPythonTools.api import Api

from tqdm import tqdm
# local imports
import constants
from common.models import CameraProjection
from osc_discoverer import Sequence
from visual_data_discover import Photo, Video
from osc_api_config import OSCAPISubDomain
from login_controller import LoginController
from osc_api_models import OSCPhoto, OSCSequence

LOGGER = logging.getLogger('osc_uploader')
THREAD_LOCK = threading.Lock()

osm_api = Api()

class Coordinates:
    def __init__(self, lat: float, long: float):
        self.long = long
        self.lat = lat

    def __len__(self):
        return 2

    def __getitem__(self, item):
        return self.long if item == 0 else self.lat

    def __repr__(self):
        return f"[{self.lat}, {self.long}]"


class OSCDownloadManager(threading.Thread):
    def __init__(self, login_controller: LoginController = None,
                 coordinates: List[Coordinates] = None, max_workers: int = 10,
                 photo_dir: str = "downloads", search_radius: int = 10,
                 show_pbar: bool = True):
        super(OSCDownloadManager, self).__init__()
        self.progress_bar = None
        self.coordinates: List[Coordinates] = coordinates if coordinates else []
        self.visual_data_count = 0
        self.login_controller = login_controller if login_controller \
            else LoginController(OSCAPISubDomain.PRODUCTION)
        self.max_workers = max_workers
        self.photo_dir = photo_dir
        self.search_radius = search_radius
        os.makedirs(photo_dir, exist_ok=True)
        self.download_complete = threading.Event()
        self.coords_path_mapping = None
        self.show_pbar = show_pbar
        self.progress_bar = None

    def add_single_coordinates(self, coords: Coordinates):
        self.coordinates.append(coords)

    def add_list_coordinates(self, coords: List[Coordinates]):
        self.coordinates += coords

    def start_download(self):
        if self.show_pbar:
            self.progress_bar = tqdm(total=len(self.coordinates), dynamic_ncols=True,
                                     desc="Querying metadata for coordinates")

        query_handler = PhotoQueryHandler(self, self.coordinates, self.search_radius,
                                          self.max_workers)
        coords_photos = query_handler.query()
        if self.progress_bar:
            self.progress_bar.close()

        osc_photos = [photo for item in coords_photos for photo in item["photos"]]

        if not osc_photos:
            LOGGER.error(msg="Could not find any photos for your coordinates.")
            return

        if self.show_pbar:
            self.progress_bar = tqdm(total=len(osc_photos), dynamic_ncols=True,
                                     desc="Downloading photos")

        download_handler = PhotoDownloadHandler(self, osc_photos, self.max_workers)

        photo_path_mappings = download_handler.download()

        if self.progress_bar:
            self.progress_bar.close()

        failed_photos = [pair["photo"] for pair in photo_path_mappings
                         if not pair["path"]]

        LOGGER.warning(msg=f"Finished download. Photos saved to {self.photo_dir}.")
        if failed_photos:
            failed_photos_string = [
                f"{photo.image_name} ({photo.longitude}, {photo.latitude})\n"
                for photo in failed_photos
            ]
            LOGGER.warning(
                msg="Could not download following images:\n"
                    f"{failed_photos_string}."
            )

        photo_path_dict = {item["photo"]: item["path"] for item in photo_path_mappings}
        self.coords_path_mapping = {
            item["coordinates"]: [photo_path_dict[photo] for photo in item["photos"]]
            for item in coords_photos
        }

    def run(self) -> None:
        self.start_download()
        self.download_complete.set()


class PhotoQueryHandler:
    def __init__(self, manager: OSCDownloadManager, coordinates: List[Coordinates],
                 radius: int = 10, max_workers: int = 10):
        self.manager = manager
        self.coordinates = coordinates
        self.radius = radius
        self.max_workers = max_workers

    def _get_coordinate_splits(self):
        total = len(self.coordinates)
        split_size = math.ceil(total / self.max_workers)
        splits = []
        for n in range(self.max_workers):
            splits.append(self.coordinates[n*split_size: (n+1)*split_size])

        return splits

    def query(self):
        splits = self._get_coordinate_splits()

        query_operations = [
            MultiplePhotoQueryOperation(self.manager, coords, self.radius)
            for coords in splits
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executors:
            futures = [executors.submit(query_operation.query)
                       for query_operation in query_operations]
            coords_photos = []
            for future in as_completed(futures):
                coords_photos += future.result()

        return coords_photos


class MultiplePhotoQueryOperation:
    def __init__(self, manager: OSCDownloadManager, coordinates: List[Coordinates],
                 radius: int = 10):
        self.manager = manager
        self.coordinates = coordinates
        self.radius = radius

    def query(self):
        api = self.manager.login_controller.osc_api

        coords_photos = []
        for coord in self.coordinates:
            osc_photos, status = api.get_nearby_photos(
                lat=coord.lat, long=coord.long, radius=self.radius
            )
            coords_photos.append({"coordinates": coord, "photos": osc_photos,
                                  "status": status})

            if self.manager.progress_bar:
                with THREAD_LOCK:
                    self.manager.progress_bar.update(1)

        return coords_photos


class PhotoDownloadHandler:
    def __init__(self, manager: OSCDownloadManager, osc_photos: List[OSCPhoto],
                 max_workers: int = 10):
        self.manager = manager
        self.osc_photos = osc_photos
        self.max_workers = max_workers

    def _get_photo_splits(self):
        total = len(self.osc_photos)
        split_size = math.ceil(total / self.max_workers)
        splits = []
        for n in range(self.max_workers):
            splits.append(self.osc_photos[n * split_size: (n + 1) * split_size])

        return splits

    def download(self):
        splits = self._get_photo_splits()

        download_operations = [
            MultiplePhotoDownloadOperation(self.manager, osc_photos)
            for osc_photos in splits
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executors:
            futures = [executors.submit(download_operation.download)
                       for download_operation in download_operations]
            photo_path_mappings = []
            for future in as_completed(futures):
                photo_path_mappings += future.result()

        return photo_path_mappings


class MultiplePhotoDownloadOperation:
    def __init__(self, manager: OSCDownloadManager, osc_photos: List[OSCPhoto]):
        self.manager = manager
        self.osc_photos = osc_photos

    def download(self):
        api = self.manager.login_controller.osc_api
        results = []
        for photo in self.osc_photos:
            dir = os.path.join(self.manager.photo_dir,
                               f"{photo.longitude}_{photo.latitude}")
            file = photo.image_name.split("/")[-1]
            photo_file_path = os.path.join(dir, file)
            os.makedirs(dir, exist_ok=True)

            download_success = api.download_photo(
                photo, out_name=photo_file_path
            )

            if not download_success:
                LOGGER.error(msg=f"Could not download {photo.image_name}. ("
                                 f"{photo.latitude}, {photo.longitude})")
                results.append({"photo": photo, "path": None})

            else:
                results.append({"photo": photo, "path": photo_file_path})

            if self.manager.progress_bar:
                with THREAD_LOCK:
                    self.manager.progress_bar.update(1)

        return results


if __name__ == "__main__":
    login_controller = LoginController(OSCAPISubDomain.PRODUCTION)
    coords = [Coordinates(34.117137, -117.858287)]
    # coords = [Coordinates(41.8795, -87.6246)]
    manager = OSCDownloadManager(login_controller=login_controller,
                                 coordinates=coords, search_radius=10)
    manager.start_download()
    a = 0
