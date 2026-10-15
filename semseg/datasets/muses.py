from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import random
import struct

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

__all__ = ["MUSES", "ORDER", "PIXEL_MEAN", "PIXEL_STD"]

ORDER = ("CAMERA", "EVENT_CAMERA", "LIDAR", "RADAR")
FOLDERS = ("frame_camera", "event_camera", "lidar", "radar")
NATIVE_HW = (1080, 1920)
PROTOCOL_HW = (1024, 1820)
BAD_LIDAR_IDS = frozenset({"REC0012_frame_218762", "REC0122_frame_206141",
                           "REC0127_frame_212950", "REC0136_frame_226214"})
PIXEL_MEAN = {"CAMERA": (123.675, 116.280, 103.530),
              "EVENT_CAMERA": (0.12577528, 0.12728328, 0.),
              "LIDAR": (4.91737, 6.149373, 0.27025607),
              "RADAR": (4.003564, 9.342789, 0.)}
PIXEL_STD = {"CAMERA": (58.395, 57.120, 57.375),
             "EVENT_CAMERA": (0.54420582, 0.47625199, 1.),
             "LIDAR": (13.34715295, 17.63751258, 1.23208644),
             "RADAR": (14.60685343, 19.29406236, 1.)}
CLASSES = ("road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
           "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
           "truck", "bus", "train", "motorcycle", "bicycle")
PALETTE = ((128,64,128), (244,35,232), (70,70,70), (102,102,156), (190,153,153),
           (153,153,153), (250,170,30), (220,220,0), (107,142,35), (152,251,152),
           (70,130,180), (220,20,60), (255,0,0), (0,0,142), (0,0,70),
           (0,60,100), (0,80,100), (0,0,230), (119,11,32))


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Require a nonempty portable relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path


def _within(base, relative):
    base = Path(base).resolve()
    path = (base / Path(relative)).resolve(strict=True)
    if base not in path.parents or not path.is_file():
        raise RuntimeError(f"Missing/unsafe data file: {path}")
    return path


def _png_shape(path):
    with Path(path).open("rb") as stream:
        header = stream.read(33)
    if len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"Invalid PNG header: {path}")
    return struct.unpack(">IIBB", header[16:26])  # width,height,bit-depth,color-type


def _check_png(path, modality):
    shape = _png_shape(path)
    if modality == "CAMERA" or modality == "GT_COLOR":
        allowed = ((1920, 1080, 8, 2),)
    elif modality in ("LIDAR", "RADAR"):
        allowed = ((1024, 1024, 16, 2),)
    elif modality == "EVENT_CAMERA":
        allowed = ((1024, 1024, 8, 2), (1024, 1024, 8, 6))
    elif modality == "GT":
        allowed = ((1920, 1080, 8, 0), (1920, 1080, 8, 3))
    else:
        raise ValueError(f"Unknown PNG modality: {modality}")
    if shape not in allowed:
        hint = " Supply the verified portable four-LiDAR overlay manifest; uint8 fallback is forbidden." if modality == "LIDAR" else ""
        raise RuntimeError(f"Unsupported physical encoding {modality}: {shape}, {path}.{hint}")


def _dataset_splits(meta):
    if not isinstance(meta, dict) or any(not isinstance(v, dict) for v in meta.values()):
        raise ValueError("meta.json must map MUSES sample IDs to metadata objects")
    if any(not isinstance(sid, str) or "/" in sid or "\\" in sid or sid in ("", ".", "..") for sid in meta):
        raise ValueError("Invalid metadata sample ID")
    if any(v.get("split") not in ("train", "val", "test") for v in meta.values()):
        raise ValueError("Unknown or missing MUSES split; no samples skipped")
    splits = {name: sorted(sid for sid, row in meta.items() if row["split"] == name)
              for name in ("train", "val", "test")}
    if {name: len(ids) for name, ids in splits.items()} != {"train": 1500, "val": 250, "test": 750}:
        raise RuntimeError("Require all MUSES 1500/250/750 metadata IDs; no subset or exclusion")
    return splits


def _canonical_modals(modals):
    aliases = {"camera": "CAMERA", "img": "CAMERA", "image": "CAMERA", "rgb": "CAMERA",
               "event_camera": "EVENT_CAMERA", "event": "EVENT_CAMERA", "lidar": "LIDAR", "radar": "RADAR"}
    if not isinstance(modals, (list, tuple)) or not all(isinstance(x, str) for x in modals):
        raise TypeError("modals must be four ordered modality names")
    normalized = tuple(aliases.get(x.lower(), x.upper()) for x in modals)
    if normalized != ORDER:
        raise ValueError(f"This checkpoint requires four slots in{ORDER}; REL still keeps normalized physical-zero RADAR, not three model inputs")
    return normalized


def _decode_projected(raw, modality):
    """Decode physical units without rescaling to [0, 1] or swapping sensor channels."""
    if modality == "CAMERA":
        if raw.dtype != np.uint8 or raw.shape != (1080, 1920, 3):
            raise RuntimeError("Camera must be native1080x1920 uint8 three-channel PNG")
        return raw[:, :, ::-1].copy().astype(np.float32)
    if modality in ("LIDAR", "RADAR"):
        if raw.dtype != np.uint16 or raw.shape != (1024, 1024, 3):
            raise RuntimeError(f"{modality} requires true1024x1024x3 uint16; no uint8/gray/alpha fallback")
        return raw.astype(np.float32) / 150. - 100.
    if modality == "EVENT_CAMERA":
        if raw.dtype != np.uint8 or raw.shape not in ((1024,1024,3), (1024,1024,4)):
            raise RuntimeError("Event requires1024x1024 uint8 with3 channels or constant opaque alpha")
        if raw.shape[2] == 4:
            if not np.all(raw[:, :, 3] == 255):
                raise RuntimeError("Event alpha is not constant opaque255; no silent removal")
            raw = raw[:, :, :3]
        return raw.astype(np.float32)
    raise ValueError(f"Unknown physical modality: {modality}")


class MUSES(Dataset):
    CLASSES = list(CLASSES)
    PALETTE = torch.tensor(PALETTE, dtype=torch.uint8)
    n_classes = 19
    ignore_label = 255
    supports_msf = False

    def __init__(self, root, split="train", transform=None, modals=ORDER, case=None, *,
                 image_size=PROTOCOL_HW, flip_prob=0.5,
                 overlay_manifest=None, input_mode="REL"):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train/val/test from MUSES meta.json")
        if case is not None:
            raise ValueError("MUSES uses complete metadata splits; case filtering is unsupported")
        if input_mode not in ("REL", "FULL"):
            raise ValueError("input_mode must be explicit REL or FULL")
        if transform is not None:
            raise TypeError("MUSES owns geometry and normalization; use MUSES.from_config or image_size, not a generic transform")
        if (not isinstance(image_size, (tuple, list)) or len(image_size) != 2
                or any(type(x) is not int or x <= 0 for x in image_size)):
            raise ValueError("image_size must contain positive integer height,width")
        if not isinstance(flip_prob, (int, float)) or not math.isfinite(flip_prob) or not 0 <= flip_prob <= 1:
            raise ValueError("flip_prob must be finite within0..1")
        self.image_size = tuple(image_size)
        self.flip_prob = float(flip_prob) if split == "train" else 0.
        if split != "train" and self.image_size != PROTOCOL_HW:
            raise ValueError("Validation/test protocol requires exactly1024x1820 inputs")
        self.root = Path(root).expanduser().resolve(strict=True)
        self.split, self.input_mode = split, input_mode
        self.modals = list(modals)
        self.slot_order = _canonical_modals(modals)
        self.meta_path = _within(self.root, PurePosixPath("meta.json"))
        meta = json.loads(self.meta_path.read_text())
        self.split_ids = _dataset_splits(meta)
        self.files = self.split_ids[split]
        self.sample_ids = list(self.files)
        self._meta = meta
        self._overlay = {}
        self._overlay = self._load_overlay(overlay_manifest)
        self.entries = [self._entry(sid, check_files=True) for sid in self.files]
        self.protocol = {"split": split, "split_counts": {k: len(v) for k,v in self.split_ids.items()},
                         "slot_order": list(ORDER), "input_mode": input_mode,
                         "genuine_three_input_model": False,
                         "missing_radar_rule": "physical zero BEFORE normalization" if input_mode == "REL" else "real projected radar",
                         "input_size_hw": list(self.image_size), "GT_size": "resized nearest" if split == "train" else "native1080x1920",
                         "normalization": "MUSES PIXEL_MEAN/PIXEL_STD; no /255 normalization",
                         "metadata_sha256": _sha256(self.meta_path),
                         "overlay_manifest_sha256": self.overlay_manifest_sha256,
                         "repaired_lidar_ids": sorted(self._overlay), "samples_skipped": 0}

    @classmethod
    def from_config(cls, dataset_cfg, split, image_size, case=None):
        """Construct the dataset without external geometry or integration helpers."""
        if dataset_cfg.get("IGNORE_LABEL", cls.ignore_label) != cls.ignore_label:
            raise ValueError("MUSES requires ignore label 255")
        return cls(dataset_cfg["ROOT"], split, modals=dataset_cfg["MODALS"], case=case,
                   image_size=image_size,
                   overlay_manifest=dataset_cfg.get("OVERLAY_MANIFEST"),
                   input_mode=dataset_cfg.get("INPUT_MODE", "REL"))

    def _apply_geometry(self, images, label):
        if len(images) != 4 or label.shape != NATIVE_HW:
            raise RuntimeError("Geometry requires four aligned native inputs and native GT")
        images = [F.interpolate(x, self.image_size, mode="bilinear", align_corners=False) for x in images]
        if self.split == "train":
            label = F.interpolate(label[None, None].float(), self.image_size,
                                  mode="nearest")[0, 0].long()
            if random.random() < self.flip_prob:
                images = [x.flip(-1) for x in images]
                label = label.flip(-1)
        return images, label

    def _entry(self, sid, check_files):
        row = self._meta[sid]
        relative = _safe_relative(row["path_to_frame_camera"])
        if len(relative.parts) < 3 or relative.parts[0] != "frame_camera" or relative.parts[1] != row["split"]:
            raise RuntimeError(f"Unexpected MUSES camera path/split for {sid}")
        if relative.name != f"{sid}_frame_camera.png":
            raise RuntimeError(f"Camera filename does not identify sample{sid}")
        tail = PurePosixPath(*relative.parts[1:-1])
        paths = {name: self.root / "multimodal" / folder / Path(tail) / f"{sid}_{folder}.png"
                 for name,folder in zip(ORDER,FOLDERS)}
        if sid in self._overlay:
            paths["LIDAR"] = self._overlay[sid]["path"]
        gt_dir = self.root / "multimodal" / "gt_semantic" / Path(tail)
        entry = {"sample_id": sid, "paths": paths,
                 "gt": gt_dir / f"{sid}_gt_labelTrainIds.png",
                 "gt_color": gt_dir / f"{sid}_gt_labelColor.png"}
        if check_files:
            for name,path in list(paths.items()) + [("GT",entry["gt"]), ("GT_COLOR",entry["gt_color"])]:
                path = path.resolve(strict=True)
                base = self.overlay_base if name == "LIDAR" and sid in self._overlay else self.root
                if base not in path.parents or not path.is_file():
                    raise RuntimeError(f"Missing/unsafe sample file{sid}/{name}: {path}")
                _check_png(path,name)
            entry["paths"] = {name: path.resolve(strict=True) for name,path in paths.items()}
        return entry

    def _load_overlay(self, manifest):
        self.overlay_manifest_sha256 = None
        self.overlay_base = None
        if manifest is None:
            return {}
        manifest = Path(manifest).expanduser().resolve(strict=True)
        body = json.loads(manifest.read_text())
        if (body.get("schema") != "crossweaver_muses_overlay_v1" or body.get("status") != "PASS"
                or body.get("projection_verified") is not True):
            raise RuntimeError("Require PASS verified portable crossweaver_muses_overlay_v1 manifest")
        rows = body.get("repairs")
        if (not isinstance(rows,list) or len(rows) != 4 or any(not isinstance(row,dict) for row in rows)
                or {row.get("sample_id") for row in rows} != BAD_LIDAR_IDS):
            raise RuntimeError("Overlay must contain exactly the four known original LiDAR IDs")
        self.overlay_base = manifest.parent
        result = {}
        for row in rows:
            sid = row["sample_id"]
            if sid not in self._meta or self._meta[sid]["split"] != "val":
                raise RuntimeError("Overlay must retain the original validation sample identity")
            relative = _safe_relative(row["relative_path"])
            if relative.parts[0] != "lidar" or relative.name != f"{sid}_lidar.png":
                raise RuntimeError("Overlay file must be the corresponding projected LiDAR PNG")
            recovered = _within(self.overlay_base,relative)
            original = self._entry(sid,check_files=False)["paths"]["LIDAR"].resolve(strict=True)
            camera_relative = _safe_relative(self._meta[sid]["path_to_frame_camera"])
            expected_relative = PurePosixPath("lidar", *camera_relative.parts[1:-1], f"{sid}_lidar.png")
            if relative != expected_relative or recovered == original:
                raise RuntimeError("Overlay must match the exact metadata-derived LiDAR path and remain a separate file")
            if self.root not in original.parents or not original.is_file():
                raise RuntimeError("Original LiDAR data path must exist and remain inside dataset root")
            for key in ("original_sha256", "recovered_sha256"):
                value = row.get(key)
                if not isinstance(value,str) or len(value) != 64 or any(x not in "0123456789abcdef" for x in value):
                    raise RuntimeError(f"Invalid manifest SHA256:{key}")
            recovered_sha = _sha256(recovered)
            original_sha = _sha256(original)
            if recovered_sha != row["recovered_sha256"] or original_sha not in (row["original_sha256"], recovered_sha):
                raise RuntimeError(f"Original/recovered LiDAR SHA differs for{sid}")
            _check_png(recovered,"LIDAR")
            result[sid] = {"path": recovered, "sha256": recovered_sha,
                           "original_path": original, "original_observed_sha256": original_sha,
                           "original_already_correct": original_sha == recovered_sha}
        self.overlay_manifest_sha256 = _sha256(manifest)
        return result

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        entry = self.entries[index]
        sid = entry["sample_id"]
        if sid in self._overlay and _sha256(entry["paths"]["LIDAR"]) != self._overlay[sid]["sha256"]:
            raise RuntimeError("Verified overlay bytes changed after dataset construction")
        with Image.open(entry["gt"]) as image:
            gt = np.array(image)
        if (gt.dtype != np.uint8 or gt.shape != NATIVE_HW
                or not set(np.unique(gt)).issubset(set(range(19)) | {255})):
            raise RuntimeError("GT must contain MUSES train IDs 0..18 or ignore label 255")
        with Image.open(entry["gt_color"]) as image:
            color = np.array(image.convert("RGB"))
        valid = gt != 255
        palette = np.asarray(PALETTE,dtype=np.uint8)
        if color.shape != (*NATIVE_HW,3) or not np.array_equal(color[valid],palette[gt[valid]]):
            raise RuntimeError("Native GT trainID/color-palette identity failed")
        inputs = []
        for name in ORDER:
            path = entry["paths"][name]
            raw = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(f"Cannot decode{sid}/{name}:{path}")
            array = _decode_projected(raw,name)
            if name == "RADAR" and self.input_mode == "REL":
                array = np.zeros_like(array,dtype=np.float32)
            x = torch.from_numpy(np.ascontiguousarray(array.transpose(2,0,1))).unsqueeze(0)
            if x.shape[-2:] != NATIVE_HW:
                x = F.interpolate(x,NATIVE_HW,mode="bilinear",align_corners=False)
            inputs.append(x)
        label = torch.from_numpy(gt.astype(np.int64))
        inputs,label = self._apply_geometry(inputs,label)
        normalized = []
        for name,x in zip(ORDER,inputs):
            mean = torch.tensor(PIXEL_MEAN[name],dtype=torch.float32).view(1,3,1,1)
            std = torch.tensor(PIXEL_STD[name],dtype=torch.float32).view(1,3,1,1)
            x = (x-mean)/std
            if not torch.isfinite(x).all():
                raise RuntimeError(f"Nonfinite MUSES normalized input: {sid}/{name}")
            if name == "RADAR" and self.input_mode == "REL" and not torch.equal(x,(-mean/std).expand_as(x)):
                raise RuntimeError("REL Radar must be normalized physical-zero, not normalized-space zero")
            normalized.append(x[0].contiguous())
        return normalized,label.long().contiguous()

    @staticmethod
    @contextmanager
    def _inference_precision(device):
        previous = (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32,
                    torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.autocast(device_type=torch.device(device).type, enabled=False):
                yield
        finally:
            (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32,
             torch.backends.cuda.matmul.allow_tf32) = previous

    @torch.no_grad()
    def evaluate(self, model, dataloader, device):
        if dataloader.dataset is not self or self.split == "train":
            raise ValueError("Native MUSES evaluation requires this complete val/test dataset")
        model.eval()
        # Validation runs on rank zero; avoid DDP forward collectives there.
        inference_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in inference_model.parameters()):
            raise TypeError("Native MUSES evaluation requires FP32 model parameters")
        hist = np.zeros((19, 19), dtype=np.int64)
        seen = 0
        with self._inference_precision(device):
            for images, labels in dataloader:
                inputs = [x.to(device, non_blocking=True) for x in images]
                logits = inference_model(inputs).float()
                if logits.ndim != 4 or logits.shape[1] != 19 or not torch.isfinite(logits).all():
                    raise RuntimeError("Invalid MUSES model logits")
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                predictions = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
                truth = labels.cpu().numpy().astype(np.int64)
                if predictions.shape != truth.shape:
                    raise RuntimeError("Native MUSES prediction/GT shape mismatch")
                valid = truth != self.ignore_label
                if ((truth[valid] < 0) | (truth[valid] >= 19)).any():
                    raise ValueError("GT contains non-native MUSES train IDs")
                hist += np.bincount(truth[valid] * 19 + predictions[valid], minlength=361).reshape(19, 19)
                seen += truth.shape[0]
        if seen != len(self) or not hist.sum():
            raise RuntimeError("MUSES evaluation must cover every image exactly once")
        tp = np.diag(hist).astype(np.float64)
        gt = hist.sum(1)
        pred = hist.sum(0)
        union = gt + pred - tp
        iou = np.divide(tp, union, out=np.zeros(19, dtype=np.float64), where=union > 0)
        acc = np.divide(tp, gt, out=np.zeros(19, dtype=np.float64), where=gt > 0)
        f1 = np.divide(2 * tp, gt + pred, out=np.zeros(19, dtype=np.float64), where=(gt + pred) > 0)
        self.last_evaluation = {
            "images": seen,
            "global_confusion_int64": hist.tolist(),
            "miou_percent_unrounded": float(iou.mean() * 100),
            "input_mode": self.input_mode,
            "protocol": "nativeGT_logits_bilinear_align_corners_false_argmax_19class_int64_FP64",
        }
        return ((acc * 100).round(2).tolist(), round(float(acc.mean() * 100), 2),
                (f1 * 100).round(2).tolist(), round(float(f1.mean() * 100), 2),
                (iou * 100).round(2).tolist(), round(float(iou.mean() * 100), 2))
