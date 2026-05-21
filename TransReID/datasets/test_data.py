import glob
import os
import os.path as osp
import random

from .bases import BaseImageDataset


class TestData(BaseImageDataset):
    """Move evaluation-only dataset.

    The original directory layout is preserved. Each person directory provides
    one deterministic random query image and all remaining images as gallery.
    """

    dataset_dir = "test"

    def __init__(self, root="", verbose=True, seed=1, **kwargs):
        super(TestData, self).__init__()
        self.dataset_dir = self._resolve_dataset_dir(root)
        self.seed = seed

        self._check_before_run()
        query, gallery = self._process_dir()

        if verbose:
            print("=> Move loaded")
            self.print_dataset_statistics([], query, gallery)

        self.train = None
        self.query = query
        self.gallery = gallery

        cams = {camid for _, _, camid, _ in self.query + self.gallery}
        views = {trackid for _, _, _, trackid in self.query + self.gallery}

        self.num_train_pids = self.num_query_pids = len({pid for _, pid, _, _ in self.query})
        self.num_train_imgs = 0
        self.num_train_cams = len(cams)
        self.num_train_vids = len(views)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
        
        self.num_train_cams = 6
        self.num_train_pids = 751

    def _resolve_dataset_dir(self, root):
        return osp.join(root, self.dataset_dir)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not self._contains_person_dirs(self.dataset_dir):
            raise RuntimeError("'{}' contains no person directories".format(self.dataset_dir))

    def _contains_person_dirs(self, path):
        return osp.isdir(path) and any(
            osp.isdir(osp.join(path, name)) for name in os.listdir(path)
        )

    def _process_dir(self):
        rng = random.Random(self.seed)
        query = []
        gallery = []

        person_dirs = sorted(
            name for name in os.listdir(self.dataset_dir)
            if osp.isdir(osp.join(self.dataset_dir, name))
        )
        fallback_pid = 0
        for person_dir in person_dirs:
            person_path = osp.join(self.dataset_dir, person_dir)
            img_paths = sorted(
                glob.glob(osp.join(person_path, "*.jpg"))
                + glob.glob(osp.join(person_path, "*.jpeg"))
                + glob.glob(osp.join(person_path, "*.png"))
                + glob.glob(osp.join(person_path, "*.bmp"))
            )
            if len(img_paths) < 2:
                raise RuntimeError("'{}' needs at least two images for query/gallery split".format(person_path))

            try:
                pid = int(person_dir)
            except ValueError:
                pid = fallback_pid
                fallback_pid += 1

            query_img = rng.choice(img_paths)
            query.append((query_img, pid, 0, 0))
            for img_path in img_paths:
                if img_path != query_img:
                    gallery.append((img_path, pid, 1, 0))

        return query, gallery