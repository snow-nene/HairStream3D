import os
import json
import shutil
import subprocess

import cv2
import numpy as np

from tqdm import tqdm
from skimage.transform import resize

from lib.options import BaseOptions

# SAM3 worker 脚本（运行在 sam3 自己的 pixi 环境中）
_SAM3_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3_seg_worker.py")


def pad_and_resize(img, width=512):
    img = np.array(img)

    if img.shape[0] > img.shape[1]:
        img_pad = np.zeros((img.shape[0], int((img.shape[0] - img.shape[1]) / 2), img.shape[2]))
        padded_img = np.concatenate((img_pad, img, img_pad), axis=1)
    else:
        img_pad = np.zeros((int((img.shape[1] - img.shape[0]) / 2), img.shape[1], img.shape[2]))
        padded_img = np.concatenate((img_pad, img, img_pad), axis=0)

    padded_img = resize(padded_img, (width, width), preserve_range=True).astype(np.uint8)

    padded_img = cv2.merge([padded_img[:, :, 0], padded_img[:, :, 1], padded_img[:, :, 2]])

    return padded_img


def pad_for_sam3(img, max_size=2048):
    """保持原始细节做正方形 padding，仅在超出上限时缩小。"""
    img = np.asarray(img)
    height, width = img.shape[:2]
    side = max(height, width)
    pad_top = (side - height) // 2
    pad_bottom = side - height - pad_top
    pad_left = (side - width) // 2
    pad_right = side - width - pad_left
    padded = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0,
    )
    if max_size > 0 and side > max_size:
        padded = cv2.resize(
            padded, (max_size, max_size), interpolation=cv2.INTER_AREA
        )
    return padded


def write_mask_to_folder(mask, filename):
    if mask.shape[0] == 3:
        mask = mask.transpose(1, 2, 0)
        mask = ((mask[:, :, 0] + mask[:, :, 1] + mask[:, :, 2]) > 0)[:, :, None]
        mask = np.concatenate([mask, mask, mask], axis=2)
    else:
        mask = mask.transpose(1, 2, 0) > 0
    cv2.imwrite(filename, mask * 255)


def _run_sam3_backend(opt, sam3_input_path, output_seg_path, output_body_path, targets):
    """通过 subprocess 调用 sam3 独立 pixi 环境中的 worker 完成分割。"""
    sam3_root = os.path.abspath(opt.sam3_root)
    sam3_python = os.path.join(sam3_root, ".pixi", "envs", "default", "bin", "python")
    checkpoint = os.path.abspath(opt.checkpoint_sam3)

    if not os.path.isfile(sam3_python):
        raise RuntimeError(
            f"找不到 SAM3 独立环境解释器: {sam3_python}\n"
            f"请先在 {sam3_root} 中执行 `pixi install`，或改用 --seg_backend sam")
    if not os.path.isfile(checkpoint):
        raise RuntimeError(
            f"找不到 SAM3 checkpoint: {checkpoint}\n"
            f"请通过 --checkpoint_sam3 指定，或改用 --seg_backend sam")

    items = [{
        "input": os.path.join(sam3_input_path, t),
        "seg_out": os.path.join(output_seg_path, t),
        "body_out": os.path.join(output_body_path, t),
    } for t in targets]
    manifest = os.path.join(output_seg_path, "_sam3_manifest.json")
    with open(manifest, "w") as f:
        json.dump({"items": items}, f, indent=2)

    cmd = [
        sam3_python, _SAM3_WORKER,
        "--manifest", manifest,
        "--checkpoint", checkpoint,
        "--hair_prompts", *[str(p) for p in opt.sam3_hair_prompts],
        "--body_prompt", opt.sam3_body_prompt,
        "--person_selection", opt.sam3_person_selection,
        "--conf_threshold", str(opt.sam3_conf_threshold),
        "--output_size", str(opt.loadSize),
        "--mask_channels", "1",   # 与旧版 SAM 一致: seg 写单通道 0/255
        "--device", opt.device,
    ]
    print("running SAM3 worker:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(manifest):
            os.remove(manifest)


def _run_sam_backend(opt, resized_path, output_seg_path, output_body_path, targets):
    """旧版 SAM ViT-H 点提示分割（保留作为 fallback）。"""
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[opt.model_type_sam](checkpoint=opt.checkpoint_sam)
    _ = sam.to(device=opt.device)
    predictor = SamPredictor(sam)

    for t in tqdm(targets):
        image = cv2.imread(os.path.join(resized_path, t))
        resized_image = image

        predictor.set_image(resized_image)

        body_masks, _, _ = predictor.predict(point_coords=np.array([[255, 255]]), point_labels=np.array([1]), multimask_output=True)
        write_mask_to_folder(body_masks, os.path.join(output_body_path, t))

        hair_pt_x = np.where(np.sum(body_masks[:, :, 255], axis=0) > 0)[0][0]
        hair_pt = np.array([[255, hair_pt_x + 5]])
        body_pt = np.array([[255, 255]])
        pts = np.concatenate([hair_pt, body_pt], axis=0)
        hair_masks, _, _ = predictor.predict(point_coords=pts, point_labels=np.array([1, 0]), multimask_output=False)
        write_mask_to_folder(hair_masks, os.path.join(output_seg_path, t))


def img2masks(opt):
    print(f"segment hair mask and body mask (backend={opt.seg_backend})")

    input_path = os.path.join(opt.root_real_imgs, 'img')
    resized_path = os.path.join(opt.root_real_imgs, 'resized_img')
    output_seg_path = os.path.join(opt.root_real_imgs, 'seg')
    output_body_path = os.path.join(opt.root_real_imgs, 'body_img')
    sam3_input_path = os.path.join(opt.root_real_imgs, '_sam3_input')

    os.makedirs(resized_path, exist_ok=True)
    os.makedirs(output_seg_path, exist_ok=True)
    os.makedirs(output_body_path, exist_ok=True)
    if opt.seg_backend == 'sam3':
        os.makedirs(sam3_input_path, exist_ok=True)

    targets = []
    for t in sorted(os.listdir(input_path)):
        image = cv2.imread(os.path.join(input_path, t))
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        resized_image = pad_and_resize(image)
        cv2.imwrite(os.path.join(resized_path, t), resized_image)
        if opt.seg_backend == 'sam3':
            sam3_image = pad_for_sam3(image, opt.sam3_max_input_size)
            cv2.imwrite(os.path.join(sam3_input_path, t), sam3_image)
        targets.append(t)

    if opt.seg_backend == 'sam3':
        try:
            _run_sam3_backend(
                opt, sam3_input_path, output_seg_path, output_body_path, targets
            )
        finally:
            shutil.rmtree(sam3_input_path, ignore_errors=True)
    else:
        _run_sam_backend(opt, resized_path, output_seg_path, output_body_path, targets)


if __name__ == "__main__":
    opt = BaseOptions().parse()
    img2masks(opt)
