#!/usr/bin/env python3
"""
SAM3 头发/身体分割 worker（必须运行在 SAM3 自己的 pixi 环境中）。

本脚本由 scripts/infer_2d/img2masks.py 与 scripts/render/compute_multiview_maps.py
通过 subprocess 调用，不直接在项目主 pixi 环境中运行（SAM3 需要 Python>=3.12 /
torch>=2.5 / CUDA>=12，与项目主环境 torch2.1+cu118 不兼容）。

调用方式（由上游脚本自动完成）:
    <sam3_root>/.pixi/envs/default/bin/python scripts/infer_2d/sam3_seg_worker.py \
        --manifest <manifest.json> \
        --checkpoint <sam3.1_multiplex.pt> \
        --hair_prompts "hair" "ponytail hair and twin tails" \
        --body_prompt "person"

manifest.json 格式:
    {"items": [{"input": "x.png", "seg_out": "y.png", "body_out": "z.png" 或 null}, ...]}

也支持目录批处理模式:
    --input_dir <resized_img/> --seg_dir <seg/> [--body_dir <body_img/>]

输出约定（与旧版 SAM img2masks 保持一致）:
    --mask_channels 1 : seg 写单通道 0/255 灰度 PNG（img2masks 的 seg/ 目录用）
    --mask_channels 3 : seg 写三通道 0/255 PNG（compute_multiview_maps 用）
    body mask 始终写三通道 0/255 PNG。

后处理逻辑（closing + 自适应桥接 + 填孔）移植自
~/Projects/sam3/hair_imgs/extract_hair.py（已在真实图片上验证）。

多人场景处理（--person_selection largest，默认开启）:
    先去除近重复 person 实例，再综合画面中心、头发占比、人物面积与边界
    占用率选择主角。不同人物即使在图像平面相交也不会合并。头发实例按整体
    重叠率和距离分位数归属，只保留属于主角的头发；body mask 同样只输出主角。
    传 all 恢复旧行为（所有人的并集）。
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── monkey-patch addmm_act ────────────────────────────────────────────
# VitDet MLP 的融合 addmm+activation kernel 会无条件把输入转成 bfloat16，
# 而 SAM3.1 checkpoint 的权重是 float32，fc1 的 bf16 输出会在下游 float32
# 的 fc2 层崩溃。这里包一层，让输出 dtype 始终与输入一致。
# sam3/__init__.py 会急切导入 model_builder（其中导入 vitdet），所以必须
# 同时 patch 源模块和 vitdet 命名空间两处。
import sam3.perflib.fused as _fused_mod

_orig_addmm_act = _fused_mod.addmm_act


def _dtype_preserving_addmm_act(activation, linear, mat1):
    in_dtype = mat1.dtype
    out = _orig_addmm_act(activation, linear, mat1)
    return out.to(in_dtype)


_fused_mod.addmm_act = _dtype_preserving_addmm_act
# ────────────────────────────────────────────────────────────────────────

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 同样 patch 已被导入的 vitdet 引用（belt-and-suspenders）
import sam3.model.vitdet as _vitdet

_vitdet.addmm_act = _dtype_preserving_addmm_act

from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    generate_binary_structure,
    label,
)

from sam3_person_selection import (
    assign_hair_to_groups,
    deduplicate_masks,
    score_person_instances,
    select_primary_hair,
    select_primary_person,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def union_text_prompt_masks(processor, state, prompts, conf_note=True):
    """对多个文本提示分别推理并 union 所有实例 mask。

    Returns:
        (H, W) bool ndarray；若未检测到任何目标返回全 False。
    """
    all_masks = []
    for prompt in prompts:
        state = processor.set_text_prompt(prompt, state)
        masks = state.get("masks")
        scores = state.get("scores")
        if masks is not None and len(masks) > 0:
            # clone() 必须：下一次 set_text_prompt 会替换 state["masks"]
            all_masks.append(masks.clone())
            if conf_note:
                n = len(masks)
                score_str = ", ".join(f"{s:.2f}" for s in scores.tolist())
                print(f"      prompt '{prompt}': {n} region(s), scores=[{score_str}]")
    if not all_masks:
        h = w = None
        return None
    combined = torch.cat(all_masks, dim=0).any(dim=0).squeeze(0).cpu().numpy()
    return combined


def collect_instance_masks(processor, state, prompts, conf_note=True):
    """对每个文本提示分别推理，逐个收集实例 mask（不做 union）。

    Returns:
        list of (H, W) bool ndarray；未检测到任何目标时返回空列表。
    """
    instances = []
    for prompt in prompts:
        state = processor.set_text_prompt(prompt, state)
        masks = state.get("masks")
        scores = state.get("scores")
        if masks is not None and len(masks) > 0:
            m = masks.detach().cpu().numpy()
            if m.ndim == 4:  # (N, 1, H, W) → (N, H, W)
                m = m[:, 0]
            for i in range(m.shape[0]):
                inst = m[i] > 0.5
                if inst.any():
                    instances.append(inst)
            if conf_note:
                n = len(masks)
                score_str = ", ".join(f"{s:.2f}" for s in scores.tolist())
                print(f"      prompt '{prompt}': {n} region(s), scores=[{score_str}]")
    return instances


def bridge_and_clean(mask, do_bridge=True):
    """形态学后处理：小半径闭运算 → 自适应膨胀桥接碎片 → 填孔。"""
    struct3 = generate_binary_structure(2, 1)  # 3×3 cross

    # 1. 小半径 closing，清理毛刺级小缺口
    mask = binary_dilation(
        binary_erosion(
            binary_dilation(mask, structure=struct3, iterations=2),
            structure=struct3,
            iterations=1,
        ),
        structure=struct3,
        iterations=2,
    )

    if do_bridge:
        # 2. 自适应桥接：若仍有多个连通块（如发绳把双马尾断开），
        #    逐步膨胀直到合并，再回缩（少回缩 1 次以保留桥）
        labeled, n_components = label(mask, structure=struct3)
        if n_components > 1:
            dilated = mask.copy()
            dilate_iters = 0
            for _ in range(100):  # 安全上限
                dilated = binary_dilation(dilated, structure=struct3)
                dilate_iters += 1
                _, n = label(dilated, structure=struct3)
                if n == 1:
                    break
            erode_iters = max(0, dilate_iters - 1)
            for _ in range(erode_iters):
                dilated = binary_erosion(dilated, structure=struct3)
            mask = dilated

    # 3. 填充内部孔洞
    mask = binary_fill_holes(mask)
    return mask


def save_mask(mask, path, channels, output_size=None):
    """mask: (H, W) bool → 0/255 PNG，可在保存时对齐下游尺寸。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = (mask.astype(np.uint8) * 255)
    if output_size and arr.shape != (output_size, output_size):
        arr = np.asarray(Image.fromarray(arr).resize(
            (output_size, output_size), resample=Image.Resampling.NEAREST
        ))
    if channels == 3:
        arr = np.stack([arr] * 3, axis=-1)
    Image.fromarray(arr).save(path)


def parse_args():
    p = argparse.ArgumentParser(description="SAM3 hair/body segmentation worker")
    p.add_argument("--manifest", type=str, default=None,
                   help="JSON manifest（与 --input_dir 二选一）")
    p.add_argument("--input_dir", type=str, default=None)
    p.add_argument("--seg_dir", type=str, default=None)
    p.add_argument("--body_dir", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="SAM3 checkpoint，如 sam3.1_multiplex.pt")
    p.add_argument("--hair_prompts", nargs="+",
                   default=["hair", "ponytail hair and twin tails",
                            "bangs and hair on the front"])
    p.add_argument("--body_prompt", type=str, default="person",
                   help="身体文本提示；传空字符串则跳过 body mask")
    p.add_argument("--person_selection", type=str, default="largest",
                   choices=["largest", "all"],
                   help="largest: 综合评分选择主人物及其头发（默认）；"
                        "all: 保留图中所有人（旧行为）")
    p.add_argument("--conf_threshold", type=float, default=0.3)
    p.add_argument("--mask_channels", type=int, default=1, choices=[1, 3],
                   help="seg 输出通道数（body 恒为 3 通道）")
    p.add_argument("--no_bridge", action="store_true",
                   help="禁用自适应膨胀桥接（仅保留 closing+填孔）")
    p.add_argument("--resolution", type=int, default=1008,
                   help="Sam3Processor 内部分辨率")
    p.add_argument("--output_size", type=int, default=None,
                   help="输出 mask 的正方形边长；pipeline 使用 512 与下游几何对齐")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def collect_items(args):
    """返回 [(input_path, seg_out, body_out|None), ...]"""
    if args.manifest:
        with open(args.manifest, "r") as f:
            data = json.load(f)
        items = []
        for it in data["items"]:
            items.append((it["input"], it["seg_out"], it.get("body_out")))
        return items

    if not (args.input_dir and args.seg_dir):
        print("[ERROR] 需要 --manifest 或 (--input_dir 与 --seg_dir)", file=sys.stderr)
        sys.exit(2)
    items = []
    for fname in sorted(os.listdir(args.input_dir)):
        if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
            continue
        inp = os.path.join(args.input_dir, fname)
        stem = os.path.splitext(fname)[0]
        seg_out = os.path.join(args.seg_dir, stem + ".png")
        body_out = (os.path.join(args.body_dir, stem + ".png")
                    if args.body_dir else None)
        items.append((inp, seg_out, body_out))
    return items


def main():
    args = parse_args()
    items = collect_items(args)
    if not items:
        print("[WARN] 没有找到任何待处理图片")
        return

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[sam3_seg_worker] {len(items)} image(s), device={device}")
    print(f"[sam3_seg_worker] hair prompts: {args.hair_prompts}")
    print(f"[sam3_seg_worker] body prompt : '{args.body_prompt}'")
    print(f"[sam3_seg_worker] person selection: {args.person_selection}")

    print("[sam3_seg_worker] Loading SAM3 image model ...")
    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        enable_segmentation=True,
        load_from_HF=False,
        checkpoint_path=args.checkpoint,
    )
    processor = Sam3Processor(
        model=model,
        resolution=args.resolution,
        device=device,
        confidence_threshold=args.conf_threshold,
    )
    print("[sam3_seg_worker] Model loaded.")

    n_ok, n_nohair = 0, 0
    for inp, seg_out, body_out in tqdm(items, desc="SAM3 seg"):
        try:
            image = Image.open(inp).convert("RGB")
        except Exception as e:
            print(f"  [ERROR] 无法读取 {inp}: {e}")
            continue

        state = processor.set_image(image)
        h_img, w_img = image.size[1], image.size[0]

        # ── hair ──
        hair_instances = deduplicate_masks(
            collect_instance_masks(processor, state, args.hair_prompts)
        )

        # ── person 实例 → 综合中心、头发占比与边界质量选择主人物 ──
        primary_mask = None
        person_labels = None
        if args.person_selection == "largest":
            processor.reset_all_prompts(state)  # 原地修改，返回 None
            person_instances = deduplicate_masks(collect_instance_masks(
                processor, state, [args.body_prompt or "person"], conf_note=False
            ))
            if person_instances:
                hair_union = (np.stack(hair_instances, axis=0).any(axis=0)
                              if hair_instances
                              else np.zeros((h_img, w_img), dtype=bool))
                primary_idx = select_primary_person(person_instances, hair_union)
                primary_mask = person_instances[primary_idx].copy()
                person_labels = [
                    mask for i, mask in enumerate(person_instances)
                    if i != primary_idx
                ]
                person_scores = score_person_instances(person_instances, hair_union)
                print(f"  [INFO] {os.path.basename(inp)}: {len(person_instances)} 个人物实例，"
                      f"主角=#{primary_idx}, score={person_scores[primary_idx]:.3f}, "
                      f"面积={int(primary_mask.sum())}px")
            else:
                print(f"  [WARN] {os.path.basename(inp)}: 未检测到人物，启用中心头发回退")

        if not hair_instances:
            print(f"  [WARN] {os.path.basename(inp)}: 未检测到头发，写空 mask")
            hair_mask = np.zeros((h_img, w_img), dtype=bool)
            n_nohair += 1
        else:
            kept = hair_instances
            if args.person_selection == "largest" and primary_mask is not None:
                assign = assign_hair_to_groups(
                    hair_instances, [primary_mask] + person_labels)
                kept = [hm for hm, gi in zip(hair_instances, assign) if gi == 0]
                if len(kept) < len(hair_instances):
                    print(f"  [INFO] 主角人物过滤: 头发区域保留 "
                          f"{len(kept)}/{len(hair_instances)}")
                if not kept:
                    print("  [WARN] 主角人物未可靠匹配到头发，写空 mask")
            elif args.person_selection == "largest":
                fallback_idx = select_primary_hair(hair_instances)
                kept = [hair_instances[fallback_idx]]
                print(f"  [WARN] 人物检测失败，仅保留中心头发实例 #{fallback_idx}")
            hair_mask = bridge_and_clean(np.stack(kept, axis=0).any(axis=0),
                                         do_bridge=not args.no_bridge)
        save_mask(hair_mask, seg_out, args.mask_channels, args.output_size)

        # ── body ──
        if body_out and args.body_prompt:
            if args.person_selection == "largest":
                if primary_mask is not None:
                    body_mask = binary_fill_holes(primary_mask)
                else:
                    body_mask = np.zeros((h_img, w_img), dtype=bool)
            else:  # all：旧行为（所有人的并集）
                processor.reset_all_prompts(state)  # 原地修改，返回 None
                body_mask = union_text_prompt_masks(
                    processor, state, [args.body_prompt], conf_note=False)
                if body_mask is None or body_mask.sum() == 0:
                    print(f"  [WARN] {os.path.basename(inp)}: 未检测到身体，写空 mask")
                    body_mask = np.zeros((h_img, w_img), dtype=bool)
                else:
                    body_mask = binary_fill_holes(body_mask)
            save_mask(body_mask, body_out, 3, args.output_size)

        n_ok += 1

    print(f"[sam3_seg_worker] Done: {n_ok} succeeded, {n_nohair} without hair")


if __name__ == "__main__":
    main()
