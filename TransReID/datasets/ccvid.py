import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class CCVID(BaseImageDataset):
    """CCVID image-frame dataset adapter.

    The official split files list video folders. This adapter expands each
    listed video folder into sorted frame image samples.
    """

    dataset_dir = "CCVID"

    def __init__(self, root="", verbose=True, pid_begin=0, **kwargs):
        super(CCVID, self).__init__()
        self.dataset_dir = self._resolve_dataset_dir(root)
        self.train_list = osp.join(self.dataset_dir, "train.txt")
        self.query_list = osp.join(self.dataset_dir, "query.txt")
        self.gallery_list = osp.join(self.dataset_dir, "gallery.txt")
        self.pid_begin = pid_begin

        self._check_before_run()
        clothes2label = self._build_clothes_label_map()

        train = self._process_list(self.train_list, clothes2label, relabel=True)
        query = self._process_list(self.query_list, clothes2label, relabel=False)
        gallery = self._process_list(self.gallery_list, clothes2label, relabel=False)

        if verbose:
            print("=> CCVID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
        
        self.num_train_cams = 37
        # self.num_train_pids = 751

    def _resolve_dataset_dir(self, root):
        if osp.exists(osp.join(root, "train.txt")):
            return root
        return osp.join(root, self.dataset_dir)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        for list_path in [self.train_list, self.query_list, self.gallery_list]:
            if not osp.exists(list_path):
                raise RuntimeError("'{}' is not available".format(list_path))

    def _build_clothes_label_map(self):
        clothes = set()
        for list_path in [self.train_list, self.query_list, self.gallery_list]:
            with open(list_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        clothes.add(parts[2])
        return {label: idx for idx, label in enumerate(sorted(clothes))}

    def _process_list(self, list_path, clothes2label, relabel=False):
        entries = []
        pid_container = set()
        with open(list_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                video_path, pid, clothes_label = parts[:3]
                pid = int(pid)
                entries.append((video_path, pid, clothes_label))
                pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
        dataset = []
        for video_path, pid, clothes_label in entries:
            full_video_path = osp.join(self.dataset_dir, video_path)
            img_paths = sorted(
                glob.glob(osp.join(full_video_path, "*.jpg"))
                + glob.glob(osp.join(full_video_path, "*.jpeg"))
                + glob.glob(osp.join(full_video_path, "*.png"))
                + glob.glob(osp.join(full_video_path, "*.bmp"))
            )
            max_frames = 8
            if len(img_paths) > max_frames:
                step = max(1, len(img_paths) // max_frames)
                img_paths = img_paths[::step][:max_frames]
            if not img_paths:
                raise RuntimeError("No images found in '{}'".format(full_video_path))

            camid = self._parse_camid(video_path)
            trackid = clothes2label[clothes_label]
            if relabel:
                pid = pid2label[pid]
            pid = self.pid_begin + pid

            for img_path in img_paths:
                dataset.append((img_path, pid, camid, trackid))
        return dataset

    def _parse_camid(self, video_path):
        # match = re.search(r"session(\d+)", video_path)
        # if match:
        #     return int(match.group(1)) - 1
        return 0
