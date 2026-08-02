#!/bin/bash
# scripts/run_pipeline.sh
# 
# 一键多视角生成与重建 Pipeline 串联脚本
# 
# 用法:
#   ./scripts/run_pipeline.sh --img_id <img_id> [--stages stage1 stage2 ...]
#
# 支持分阶段执行:
#   prepare  : 输入原始单张图片，抠图并提取 2D 特征图 (strand/depth/seg)
#   pixal3d  : 通过 Pixal3D 生成初始 3D 网格 (glb)
#   extract  : 从 GLB 中对齐并提取 3D 毛发网格 (obj)
#   render   : 使用 Blender 将 3D 网格渲染为多视角图像 (front/left/right/back)
#   flux     : 使用 FLUX.2-klein 对渲染图进行发丝增强重绘
#   maps     : 提取多视角特征图 (strand_map, depth_map, seg)
#   pde      : 融合多视角并进行 3D PDE 求解，生成最终 hair_multiview.ply (默认为多视角，也可跑单视角)
#   preview  : 最终步骤，将生成的 3D 毛发网格重新渲染为 2D 预览图
# 
# 示例:
#   # 运行全部阶段 (默认)
#   bash scripts/run_pipeline.sh --img_id 0a1ba3dbefc8934ab60577c5c91f66a0
#   
#   # 仅运行后续 2 个阶段
#   bash scripts/run_pipeline.sh --img_id 0a1ba3dbefc8934ab60577c5c91f66a0 --stages maps pde

IMG_ID=""
RAW_IMG=""
STAGES=()

# 参数解析
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --img_id) IMG_ID="$2"; shift ;;
        --raw_img) RAW_IMG="$2"; shift ;;
        --stages)
            shift
            while [[ "$#" -gt 0 && ! "$1" == "--"* ]]; do
                STAGES+=("$1")
                shift
            done
            continue
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$IMG_ID" ]; then
    echo "错误: 必须提供 --img_id 参数。"
    echo "用法: $0 --img_id <id> [--stages stage1 stage2 ...]"
    exit 1
fi

if [ ${#STAGES[@]} -eq 0 ]; then
    # 如果指定了 raw_img，默认包含 prepare 阶段
    if [ -n "$RAW_IMG" ]; then
        STAGES=("prepare" "pixal3d" "extract" "align_calib" "render" "flux" "maps" "pde" "preview")
    else
        STAGES=("pixal3d" "extract" "align_calib" "render" "flux" "maps" "pde" "preview")
    fi
fi

run_stage() {
    local target=$1
    for stage in "${STAGES[@]}"; do
        if [ "$stage" == "$target" ] || [ "$stage" == "all" ]; then
            return 0
        fi
    done
    return 1
}

set -e

DATA_DIR="results/multiview_data/${IMG_ID}"
mkdir -p "$DATA_DIR"

if run_stage "prepare"; then
    echo "=================================================="
    echo " [0/5] Prepare: 抠图并提取 2D 特征图"
    echo "=================================================="
    if [ -z "$RAW_IMG" ]; then
        echo "错误: 执行 prepare 阶段必须提供 --raw_img <path> 参数。"
        exit 1
    fi
    TMP_DIR="${DATA_DIR}/tmp_2d"
    mkdir -p "${TMP_DIR}/img"
    cp "$RAW_IMG" "${TMP_DIR}/img/front.png"
    
    # 运行 2D 特征提取
    PYTHONPATH=. pixi run python scripts/infer_2d/img2hairstep.py --root_real_imgs "$TMP_DIR"
    
    # 将输出转移至规范的 maps 目录
    mkdir -p "${DATA_DIR}/maps/strand_map"
    mkdir -p "${DATA_DIR}/maps/depth_map"
    mkdir -p "${DATA_DIR}/maps/seg"
    mkdir -p "${DATA_DIR}/blender_renders"
    mkdir -p "${DATA_DIR}/maps/param"
    
    cp "${TMP_DIR}/seg/front.png" "${DATA_DIR}/maps/seg/"
    cp "${TMP_DIR}/strand_map/front.png" "${DATA_DIR}/maps/strand_map/"
    cp "${TMP_DIR}/depth_map/front.npy" "${DATA_DIR}/maps/depth_map/"
    cp "${TMP_DIR}/resized_img/front.png" "${DATA_DIR}/blender_renders/" # 作为基准正面渲染图
    
    # 保留原始图路径供 align_calib 阶段使用
    echo "$RAW_IMG" > "${DATA_DIR}/raw_img_path.txt"
    
    rm -rf "$TMP_DIR"
fi

if run_stage "pixal3d"; then
    echo "=================================================="
    echo " [1/5] Pixal3D: 从正面图生成初始 3D 网格"
    echo "=================================================="
    # 假设输入图片已被放入 blender_renders/front.png 或真实图像目录
    # 此处 run_pixal3d 会默认去尝试读取 blender_renders/front.png
    pixi run python scripts/infer_2d/run_pixal3d.py --img_id "$IMG_ID"
fi

if run_stage "extract"; then
    echo "=================================================="
    echo " [1.5/5] Extract: 提取并对齐 3D 初始网格"
    echo "=================================================="
    GLB_PATH="${DATA_DIR}/pixal3d/${IMG_ID}.glb"
    if [ ! -f "$GLB_PATH" ]; then
        echo "错误: 找不到 3D 基础网格文件 $GLB_PATH，请先完成 pixal3d 阶段。"
        exit 1
    fi
    pixi run python scripts/utils/align_and_extract_hair.py \
        --glb "$GLB_PATH" \
        --out_dir "${DATA_DIR}/pixal3d"
fi

if run_stage "align_calib"; then
    echo "=========================================="
    echo " [1.8/5] AlignCalib: lmk 对齐 → 生成相机标定 front.npy"
    echo "=========================================="
    RENDER_IMG="${DATA_DIR}/pixal3d/render_front.png"
    RENDER_CAM="${DATA_DIR}/pixal3d/render_camera.npz"
    HAIR_OBJ="${DATA_DIR}/pixal3d/hair_mesh_aligned_best.obj"
    STRAND_MAP="${DATA_DIR}/maps/strand_map/front.png"
    PARAM_OUT="${DATA_DIR}/maps/param"

    if [ ! -f "$RENDER_IMG" ] || [ ! -f "$RENDER_CAM" ]; then
        echo "错误: 找不到渲染图或相机参数 ($RENDER_IMG / $RENDER_CAM)。"
        echo "      请先完成 extract 阶段 (align_and_extract_hair.py 会生成这两个文件)。"
        exit 1
    fi

    # 读取原始输入图路径 (prepare 阶段写入)
    if [ -f "${DATA_DIR}/raw_img_path.txt" ]; then
        INPUT_IMG=$(cat "${DATA_DIR}/raw_img_path.txt")
    elif [ -n "$RAW_IMG" ]; then
        INPUT_IMG="$RAW_IMG"
    else
        # fallback: 用 blender_renders/front.png (已对齐的渲染结果)
        INPUT_IMG="${DATA_DIR}/blender_renders/front.png"
        echo "  [WARN] 未找到原始图路径，使用渲染图作为输入: $INPUT_IMG"
    fi

    mkdir -p "$PARAM_OUT"
    pixi run python scripts/utils/align_glb_lmk.py \
        --input_img  "$INPUT_IMG" \
        --render_img "$RENDER_IMG" \
        --render_cam "$RENDER_CAM" \
        --glb_ply    "$HAIR_OBJ" \
        --strand_map "$STRAND_MAP" \
        --out_dir    "$PARAM_OUT"

    # align_glb_lmk.py 输出为 glb_param.npy，重命名为 front.npy
    if [ -f "${PARAM_OUT}/glb_param.npy" ]; then
        mv "${PARAM_OUT}/glb_param.npy" "${PARAM_OUT}/front.npy"
        echo "  ✓ 相机标定已保存: ${PARAM_OUT}/front.npy"
    else
        echo "  [ERROR] align_glb_lmk.py 未生成 glb_param.npy，请检查！"
        exit 1
    fi
fi

if run_stage "render"; then
    echo "=================================================="
    echo " [2/5] Render: Blender 多视角渲染"
    echo "=================================================="
    GLB_PATH="${DATA_DIR}/pixal3d/${IMG_ID}.glb"
    if [ ! -f "$GLB_PATH" ]; then
        echo "错误: 找不到 3D 基础网格文件 $GLB_PATH，请先完成 pixal3d 阶段。"
        exit 1
    fi
    blender -b -P scripts/render/render_multiview_blender.py -- \
        --glb "$GLB_PATH" \
        --out_dir "${DATA_DIR}/blender_renders" \
        --size 512
        
    # Rename left_rgb, right_rgb, back_rgb to left, right, back so downstream scripts find them
    for v in left right back; do
        if [ -f "${DATA_DIR}/blender_renders/${v}_rgb.png" ]; then
            mv "${DATA_DIR}/blender_renders/${v}_rgb.png" "${DATA_DIR}/blender_renders/${v}.png"
        fi
    done
fi

if run_stage "flux"; then
    echo "=================================================="
    echo " [3/5] FLUX.2: 多视角发丝重绘增强"
    echo "=================================================="
    pixi run python scripts/infer_2d/flux_redraw_multiview.py \
        --img_id "$IMG_ID" \
        --views front left right back
fi

if run_stage "maps"; then
    echo "=================================================="
    echo " [4/5] Maps: 提取多视角 2D 特征图"
    echo "=================================================="
    # 提取特征需要基准的正面图片信息 (在生成多视角前需存在)
    FRONT_IMG="${DATA_DIR}/blender_renders/front.png"
    FRONT_STRAND="${DATA_DIR}/maps/strand_map/front.png"
    FRONT_DEPTH="${DATA_DIR}/maps/depth_map/front.npy"
    
    # 兼容原脚本的参数
    pixi run python scripts/render/compute_multiview_maps.py \
        --front_img "$FRONT_IMG" \
        --front_strand "$FRONT_STRAND" \
        --front_depth "$FRONT_DEPTH" \
        --render_dir "${DATA_DIR}/flux_redrawn" \
        --out_dir "${DATA_DIR}/maps" \
        --views left right back
fi

if run_stage "pde"; then
    echo "=================================================="
    echo " [5/6] PDE: 3D 融合解算 (生成毛发)"
    echo "=================================================="
    # 若要跑单视角，直接不加 --expand-multiview。在脚本这里为了演示多视角流程默认加上了。
    # 也可以专门运行 python scripts/recon_3d/run_pde_multiview.py --img_id "$IMG_ID" 跑单视角
    pixi run python scripts/recon_3d/run_pde_multiview.py \
        --img_id "$IMG_ID" \
        --pde_resolution 384 \
        --pde_dilation_iters 25 \
        --expand-multiview
fi

if run_stage "preview"; then
    echo "=================================================="
    echo " [6/6] Preview: 渲染最终 3D 毛发预览图"
    echo "=================================================="
    # 假设 PDE 输出保存在 pde_reconstruction/hair_multiview.ply
    PLY_PATH="${DATA_DIR}/pde_reconstruction/hair_multiview.ply"
    if [ ! -f "$PLY_PATH" ]; then
        echo "警告: 未找到 $PLY_PATH，跳过预览阶段。"
    else
        blender -b -P scripts/render/render_blender.py -- \
            --hair_path "$PLY_PATH" \
            --output_path "${DATA_DIR}/pde_reconstruction/preview.png"
    fi
fi

echo "=================================================="
echo " Pipeline 全部完成！输出目录: $DATA_DIR"
echo "=================================================="
