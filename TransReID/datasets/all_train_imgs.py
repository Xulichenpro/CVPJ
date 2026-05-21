# encoding: utf-8

from .bases import BaseImageDataset
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .occ_duke import OCC_DukeMTMCreID
import os.path as osp


class AllTrainImgs(BaseImageDataset):
    """Merged person ReID dataset for training.

    The merged training set contains DukeMTMC-reID, Market1501, MSMT17, and
    Occluded-Duke. PIDs, camera IDs, and view IDs are remapped so labels from
    different datasets do not collide.
    """

    dataset_names = ("dukemtmc", "market1501", "msmt17", "occ_duke")

    def __init__(self, root="", verbose=True, **kwargs):
        super(AllTrainImgs, self).__init__()

        source_factories = (
            ("dukemtmc", DukeMTMCreID),
            ("market1501", Market1501),
            ("msmt17", MSMT17),
            ("occ_duke", OCC_DukeMTMCreID),
        )

        train, query, gallery = [], [], []
        train_pid_begin = 0
        eval_pid_begin = 0
        cam_begin = 0
        view_begin = 0

        for name, dataset_factory in source_factories:
            if not self._dataset_dir_exists(root, dataset_factory):
                if verbose:
                    print(
                        "=> {} skipped: dataset directory is not available".format(
                            name
                        )
                    )
                continue
            try:
                dataset = dataset_factory(root=root, verbose=False)
            except RuntimeError as err:
                if verbose:
                    print("=> {} skipped: {}".format(name, err))
                continue

            train_part, train_pid_begin = self._remap_train_pids(
                dataset.train, train_pid_begin
            )
            query_part, gallery_part, eval_pid_begin = self._remap_eval_pids(
                dataset.query, dataset.gallery, eval_pid_begin
            )

            cam_map = self._build_label_map(
                [item[2] for item in dataset.train + dataset.query + dataset.gallery],
                cam_begin,
            )
            view_map = self._build_label_map(
                [item[3] for item in dataset.train + dataset.query + dataset.gallery],
                view_begin,
            )

            train_part = self._remap_cam_view(train_part, cam_map, view_map)
            query_part = self._remap_cam_view(query_part, cam_map, view_map)
            gallery_part = self._remap_cam_view(gallery_part, cam_map, view_map)

            train.extend(train_part)
            query.extend(query_part)
            gallery.extend(gallery_part)

            cam_begin += len(cam_map)
            view_begin += len(view_map)

            if verbose:
                print(
                    "=> {} merged: {} train ids, {} train images, {} cameras".format(
                        name,
                        dataset.num_train_pids,
                        dataset.num_train_imgs,
                        dataset.num_train_cams,
                    )
                )

        if len(train) == 0:
            raise RuntimeError(
                "No dataset is available for AllTrainImgs under root {}".format(root)
            )

        if verbose:
            print("=> AllTrainImgs loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    @staticmethod
    def _build_label_map(labels, offset):
        return {label: offset + idx for idx, label in enumerate(sorted(set(labels)))}

    def _remap_train_pids(self, data, pid_begin):
        pid_map = self._build_label_map([item[1] for item in data], pid_begin)
        remapped = [
            (img_path, pid_map[pid], camid, viewid)
            for img_path, pid, camid, viewid in data
        ]
        return remapped, pid_begin + len(pid_map)

    def _remap_eval_pids(self, query, gallery, pid_begin):
        pid_map = self._build_label_map(
            [item[1] for item in query + gallery], pid_begin
        )
        query_remapped = [
            (img_path, pid_map[pid], camid, viewid)
            for img_path, pid, camid, viewid in query
        ]
        gallery_remapped = [
            (img_path, pid_map[pid], camid, viewid)
            for img_path, pid, camid, viewid in gallery
        ]
        return query_remapped, gallery_remapped, pid_begin + len(pid_map)

    @staticmethod
    def _remap_cam_view(data, cam_map, view_map):
        return [
            (img_path, pid, cam_map[camid], view_map[viewid])
            for img_path, pid, camid, viewid in data
        ]
    
    @staticmethod
    def _dataset_dir_exists(root, dataset_factory):
        dataset_dir = getattr(dataset_factory, "dataset_dir", None)
        return dataset_dir is not None and osp.exists(osp.join(root, dataset_dir))

