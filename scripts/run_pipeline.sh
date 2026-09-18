#!/bin/bash
# scripts/run_pipeline.sh
# 
# 一键多视角生成与重建 Pipeline 串联脚本
# 
# 用法:
#   ./scripts/run_pipeline.sh --img_id <img_id> [--stages stage1 stage2 ...]
#
# 支持分阶段执行:
#   prepare     : 输入原始单张图片，抠图并提取 2D 特征图 (strand/depth/seg)
#   pixal3d     : 通过 Pixal3D 生成初始 3D 网格 (glb)
#   extract     : 从 GLB 中对齐并提取 3D 毛发网格 (obj)
#   align_calib : 原图、GLB 与 DINOv3+轮廓标定
#   render      : 使用 Blender 将 3D 网格渲染为多视角图像 (front/left/right/back)
#   flux        : 使用 FLUX.2-klein 对渲染图进行发丝增强重绘
#   maps        : 提取多视角特征图 (strand_map, depth_map, seg)
#   partition   : 可选阶段，组装可见证据与体积分区 bundle (build_volume_partition_bundle.py)
#   pde         : 融合多视角并进行 3D PDE 求解，生成最终 3D 发丝 (run_pde_multiview.py)
#   fit_head    : 适配求值 Blender 模板头模，消除穿模并约束管径间隙 (fit_strands_to_template_head.py)
#   grow_surface: 可选阶段，从标定视角可见头皮缺口增补表面发丝 (grow_visible_surface_gaps.py)
#   preview     : 渲染 3D 发丝多视角预览图 (render_all_prefixes.py / render_blender.py)
# 
# 示例:
#   # 运行全部默认阶段
#   bash scripts/run_pipeline.sh --img_id 0a1ba3dbefc8934ab60577c5c91f66a0
#
#   # 启用体积分区与表面补生长完整闭环
#   bash scripts/run_pipeline.sh --img_id 0a1ba3dbefc8934ab60577c5c91f66a0 --with-partition --with-grow-surface
#   
#   # 仅运行贴合与补生长阶段
#   bash scripts/run_pipeline.sh --img_id 0a1ba3dbefc8934ab60577c5c91f66a0 --stages fit_head grow_surface preview

IMG_ID=""
RAW_IMG=""
STAGES=()
VIEWS=("front" "left" "right" "back")
PDE_MODE="governed"
VOLUME_PARTITION_BUNDLE=""
WITH_PARTITION=false
WITH_GROW_SURFACE=false
PARTITION_SEED_GEOMETRY="mesh_raycast"
TEMPLATE_HEAD_PATH=""
SURFACE_GROWTH_VIEW="back"
SURFACE_GROWTH_BUDGET=300
SURFACE_OFFSET_M="0.0008"

# 参数解析
POSITIONAL_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --img_id) IMG_ID="$2"; shift ;;
        --raw_img) RAW_IMG="$2"; shift ;;
        --pde-mode) PDE_MODE="$2"; shift ;;
        --volume-partition-bundle) VOLUME_PARTITION_BUNDLE="$2"; shift ;;
        --legacy-pde) PDE_MODE="legacy" ;;
        --with-partition) WITH_PARTITION=true ;;
        --with-grow-surface) WITH_GROW_SURFACE=true ;;
        --partition-seed-geometry) PARTITION_SEED_GEOMETRY="$2"; shift ;;
        --template-head) TEMPLATE_HEAD_PATH="$2"; shift ;;
        --surface-growth-view) SURFACE_GROWTH_VIEW="$2"; shift ;;
        --surface-growth-budget) SURFACE_GROWTH_BUDGET="$2"; shift ;;
        --surface-offset-m) SURFACE_OFFSET_M="$2"; shift ;;
        --views)
            shift
            VIEWS=()
            while [[ "$#" -gt 0 && ! "$1" == "--"* ]]; do
                VIEWS+=("$1")
                shift
            done
            continue
            ;;
        --stages)
            shift
            while [[ "$#" -gt 0 && ! "$1" == "--"* ]]; do
                STAGES+=("$1")
                shift
            done
            continue
            ;;
        -*) echo "未知参数: $1"; exit 1 ;;
        *) POSITIONAL_ARGS+=("$1") ;;
    esac
    shift
done

# 如果通过位置参数传了路径或 ID
if [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
    ARG0="${POSITIONAL_ARGS[0]}"
    if [ -f "$ARG0" ]; then
        RAW_IMG="$ARG0"
    elif [ -f "results/real_imgs/${ARG0}" ]; then
        RAW_IMG="results/real_imgs/${ARG0}"
    elif [ -f "results/real_imgs/${ARG0}.png" ]; then
        RAW_IMG="results/real_imgs/${ARG0}.png"
    elif [ -f "results/real_imgs/${ARG0}.jpg" ]; then
        RAW_IMG="results/real_imgs/${ARG0}.jpg"
    else
        # 假设其为 IMG_ID
        if [ -z "$IMG_ID" ]; then
            IMG_ID="$ARG0"
        fi
    fi
fi

# 如果未显式提供 --img_id，但提供了 --raw_img，自动从文件名中提取 img_id
if [ -z "$IMG_ID" ] && [ -n "$RAW_IMG" ]; then
    filename=$(basename "$RAW_IMG")
    IMG_ID="${filename%.*}"
    echo "[INFO] 自动从 --raw_img 提取 img_id: $IMG_ID"
fi

# 如果已知 --img_id 但未提供 --raw_img，自动在 results/real_imgs 下查找匹配图片
if [ -n "$IMG_ID" ] && [ -z "$RAW_IMG" ]; then
    for ext in png jpg jpeg PNG JPG; do
        if [ -f "results/real_imgs/${IMG_ID}.${ext}" ]; then
            RAW_IMG="results/real_imgs/${IMG_ID}.${ext}"
            echo "[INFO] 自动推导 raw_img 路径: $RAW_IMG"
            break
        fi
    done
fi

if [ -z "$IMG_ID" ]; then
    echo "错误: 必须提供 --img_id 参数 (或提供 --raw_img 图片路径)。"
    echo "用法: $0 --img_id <id> [--raw_img <path>] [--views front left ...] [--stages stage1 ...]"
    exit 1
fi

if [ "$PDE_MODE" != "governed" ] && [ "$PDE_MODE" != "legacy" ]; then
    echo "错误: --pde-mode 只能是 governed 或 legacy。"
    exit 1
fi

if [ ${#STAGES[@]} -eq 0 ]; then
    STAGES=()
    if [ -n "$RAW_IMG" ]; then
        STAGES+=("prepare")
    fi
    STAGES+=("pixal3d" "extract" "align_calib" "render" "flux" "maps")
    if [ "$WITH_PARTITION" = true ]; then
        STAGES+=("partition")
    fi
    STAGES+=("pde" "fit_head")
    if [ "$WITH_GROW_SURFACE" = true ]; then
        STAGES+=("grow_surface")
    fi
    STAGES+=("preview")
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

# ==== 自动将原始输入图拷贝一份到数据目录 ====
# 保证各阶段都能从数据目录内取回原图 (raw_img.<ext>)，不依赖外部路径
# （外部路径可能被移动/删除），避免 front mask 被迫用网格渲染图重算。
SRC_IMG="$RAW_IMG"
if [ -z "$SRC_IMG" ] && [ -f "${DATA_DIR}/raw_img_path.txt" ]; then
    SRC_IMG=$(cat "${DATA_DIR}/raw_img_path.txt")
fi
if [ -n "$SRC_IMG" ] && [ -f "$SRC_IMG" ]; then
    EXT="${SRC_IMG##*.}"
    if [ "$EXT" = "$SRC_IMG" ]; then
        EXT="png"
    fi
    RAW_COPY="${DATA_DIR}/raw_img.${EXT}"
    if [ -n "$RAW_IMG" ] || [ ! -f "$RAW_COPY" ]; then
        cp "$SRC_IMG" "$RAW_COPY"
        echo " [INFO] 已拷贝原始输入图到 ${RAW_COPY}"
    fi
else
    echo " [WARN] 未找到原始输入图 (RAW_IMG=${RAW_IMG:-未设置})，跳过 raw_img 拷贝。"
fi

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
    
    if [ -d "${TMP_DIR}/param" ]; then
        cp -r "${TMP_DIR}/param/"* "${DATA_DIR}/maps/param/"
    fi
    
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
    echo " [1.8/5] AlignCalib: 原图、GLB 与 DINOv3+轮廓标定"
    echo "=========================================="
    PARAM_OUT="${DATA_DIR}/maps/param"
    mkdir -p "$PARAM_OUT"

    # 读取原始输入图路径
    if [ -f "${DATA_DIR}/raw_img_path.txt" ]; then
        INPUT_IMG=$(cat "${DATA_DIR}/raw_img_path.txt")
    elif [ -n "$RAW_IMG" ]; then
        INPUT_IMG="$RAW_IMG"
    fi

    # 1. 确保原始图片对应的校准矩阵 front.npy 存在且不被覆盖
    if [ ! -f "${PARAM_OUT}/front.npy" ]; then
        echo "  --> 正在生成原始输入图的校准矩阵 maps/param/front.npy..."
        if [ -n "$INPUT_IMG" ] && [ -f "$INPUT_IMG" ]; then
            # 建立 opt_cam 期望的临时图像目录
            TMP_OPT="${DATA_DIR}/tmp_opt"
            mkdir -p "${TMP_OPT}/resized_img" "${TMP_OPT}/lmk"
            cp "$INPUT_IMG" "${TMP_OPT}/resized_img/front.png"
            
            # 检测关键点并优化标定
            PYTHONPATH=. pixi run python scripts/utils/get_lmk.py --root_real_imgs "$TMP_OPT"
            PYTHONPATH=. pixi run python scripts/utils/opt_cam.py --root_real_imgs "$TMP_OPT"
            
            if [ -f "${TMP_OPT}/param/front.npy" ]; then
                cp "${TMP_OPT}/param/front.npy" "${PARAM_OUT}/front.npy"
            fi
            rm -rf "$TMP_OPT"
        fi
    else
        echo "  ✓ 原始输入图标定已存在: ${PARAM_OUT}/front.npy"
    fi

    # 2. 运行 align_glb_lmk.py 生成 Pixal3D GLB 模型专属的 glb_param.npy
    RENDER_IMG="${DATA_DIR}/pixal3d/render_front.png"
    RENDER_CAM="${DATA_DIR}/pixal3d/render_camera.npz"
    HAIR_OBJ="${DATA_DIR}/pixal3d/hair_mesh_aligned_best.obj"
    STRAND_MAP="${DATA_DIR}/maps/strand_map/front.png"

    if [ -f "$RENDER_IMG" ] && [ -f "$RENDER_CAM" ] && [ -f "$HAIR_OBJ" ]; then
        echo "  --> 正在计算 Pixal3D 模型专属标定矩阵 maps/param/glb_param.npy..."
        pixi run python scripts/utils/align_glb_lmk.py \
            --input_img  "${INPUT_IMG:-${DATA_DIR}/blender_renders/front.png}" \
            --render_img "$RENDER_IMG" \
            --render_cam "$RENDER_CAM" \
            --glb_ply    "$HAIR_OBJ" \
            --strand_map "$STRAND_MAP" \
            --out_dir    "$PARAM_OUT"

        if [ -f "${PARAM_OUT}/glb_param.npy" ]; then
            echo "  ✓ GLB 模型专属标定已保存: ${PARAM_OUT}/glb_param.npy"
        else
            echo "  [ERROR] align_glb_lmk.py 未生成 glb_param.npy，请检查！"
            exit 1
        fi
    fi

    # 3. 正式 front 几何标定：DINOv3 稠密对应估计 SO(3)，front seg
    # 与 Pixal3D 渲染头发 mask 进一步约束二维轮廓。失败时保留旧 front.npy。
    DENSE_OUT="${PARAM_OUT}/dense_front"
    DENSE_PARAM="${PARAM_OUT}/front_dense_silhouette.npy"
    echo "  --> 正在计算 DINOv3 + 头发轮廓 front 标定..."
    if pixi run python scripts/utils/fit_front_pose_dense.py \
        --img_id "$IMG_ID" \
        --image "${INPUT_IMG:-${DATA_DIR}/blender_renders/front.png}" \
        --out_dir "$DENSE_OUT" \
        --auto_render_hair_mask \
        --require_silhouette; then
        if [ -f "${DENSE_OUT}/front_dense_silhouette.npy" ]; then
            cp "${DENSE_OUT}/front_dense_silhouette.npy" "$DENSE_PARAM"
            echo "  ✓ 正式稠密 front 标定已保存: $DENSE_PARAM"
        else
            echo "  [WARN] 稠密标定未生成最终参数，PDE 将回退 maps/param/front.npy"
        fi
    else
        echo "  [WARN] DINOv3 + 轮廓标定失败，PDE 将回退 maps/param/front.npy"
    fi

    # 4. 运行对齐效果可视化，生成 real_face_alignment.png 供检查
    pixi run python scripts/utils/vis_calib_alignment.py \
        --img_id "$IMG_ID" \
        --out_dir "$PARAM_OUT"
fi

if run_stage "render"; then
    echo "=================================================="
    echo " [2/5] Render: Blender 多视角渲染 (${VIEWS[*]})"
    echo "=================================================="
    GLB_PATH="${DATA_DIR}/pixal3d/${IMG_ID}.glb"
    if [ ! -f "$GLB_PATH" ]; then
        echo "错误: 找不到 3D 基础网格文件 $GLB_PATH，请先完成 pixal3d 阶段。"
        exit 1
    fi
    blender -b --factory-startup -P scripts/render/render_multiview_blender.py -- \
        --glb "$GLB_PATH" \
        --out_dir "${DATA_DIR}/blender_renders" \
        --size 512 \
        --views "${VIEWS[@]}"
        
    # Rename *_rgb.png to standard view names
    for v in "${VIEWS[@]}"; do
        if [ -f "${DATA_DIR}/blender_renders/${v}_rgb.png" ]; then
            mv "${DATA_DIR}/blender_renders/${v}_rgb.png" "${DATA_DIR}/blender_renders/${v}.png"
        fi
    done
fi

if run_stage "flux"; then
    echo "=================================================="
    echo " [3/5] FLUX.2: 多视角发丝重绘增强 (${VIEWS[*]})"
    echo "=================================================="
    pixi run python scripts/infer_2d/flux_redraw_multiview.py \
        --img_id "$IMG_ID" \
        --views "${VIEWS[@]}"
fi

if run_stage "maps"; then
    echo "=================================================="
    echo " [4/5] Maps: 提取多视角 2D 特征图 (${VIEWS[*]})"
    echo "=================================================="
    # front 的 mask/depth 必须基于原始照片计算；blender_renders/front.png
    # 在 render 阶段后已被 3D 网格渲染图覆盖，不能用作 front 输入。
    # 优先级：数据目录内的 raw_img.* 拷贝 > raw_img_path.txt 记录的外部路径 > blender_renders/front.png
    FRONT_IMG=""
    for cand in "${DATA_DIR}"/raw_img.*; do
        if [ -f "$cand" ]; then
            FRONT_IMG="$cand"
            break
        fi
    done
    if [ -z "$FRONT_IMG" ] && [ -f "${DATA_DIR}/raw_img_path.txt" ]; then
        RAW_IMG_PATH=$(cat "${DATA_DIR}/raw_img_path.txt")
        if [ -f "$RAW_IMG_PATH" ]; then
            FRONT_IMG="$RAW_IMG_PATH"
        fi
    fi
    if [ -z "$FRONT_IMG" ]; then
        FRONT_IMG="${DATA_DIR}/blender_renders/front.png"
    fi
    FRONT_STRAND="${DATA_DIR}/maps/strand_map/front.png"
    FRONT_DEPTH="${DATA_DIR}/maps/depth_map/front.npy"
    
    # 筛选出除 front 之外的其他侧视角
    OTHER_VIEWS=()
    for v in "${VIEWS[@]}"; do
        if [ "$v" != "front" ]; then
            OTHER_VIEWS+=("$v")
        fi
    done

    if [ ${#OTHER_VIEWS[@]} -gt 0 ]; then
        pixi run python scripts/render/compute_multiview_maps.py \
            --front_img "$FRONT_IMG" \
            --front_strand "$FRONT_STRAND" \
            --front_depth "$FRONT_DEPTH" \
            --render_dir "${DATA_DIR}/flux_redrawn" \
            --out_dir "${DATA_DIR}/maps" \
            --views "${OTHER_VIEWS[@]}"
    else
        echo "  [INFO] 仅选择正面 front 视角，跳过侧面特征图提取。"
    fi
fi

if run_stage "partition"; then
    echo "=================================================="
    echo " [4.5/8] Partition: 组装体积分区 Bundle"
    echo "=================================================="
    MESH_PATH="${DATA_DIR}/pixal3d/hair_mesh_aligned_best.obj"
    if [ ! -f "$MESH_PATH" ]; then
        echo "错误: 找不到 3D 毛发网格 $MESH_PATH，请先完成 extract 阶段。"
        exit 1
    fi
    AUDIT_DIR="${DATA_DIR}/pde_governance/volume_domain_audit"
    STEP7="${AUDIT_DIR}/step_07_surface_evidence_continuity/refined_evidence.npz"
    STEP6="${AUDIT_DIR}/step_06_four_class_evidence/volume_evidence.npz"
    STEP5="${AUDIT_DIR}/step_05_closed_external_envelope/closed_outer_candidate.npz"
    if [ ! -f "$STEP7" ] || [ ! -f "$STEP6" ] || [ ! -f "$STEP5" ]; then
        echo "错误: partition 阶段需要 volume_domain_audit 前置审计产物 (step_05, step_06, step_07)。"
        echo "       未在 ${AUDIT_DIR} 下找到完整的 refined_evidence / volume_evidence / closed_outer_candidate。"
        exit 1
    fi

    BUNDLE_OUT_DIR="${DATA_DIR}/pde_governance/volume_partition_integration/bundle"
    mkdir -p "$BUNDLE_OUT_DIR"

    echo "  --> 正在调用 build_volume_partition_bundle.py 组装体积分区 bundle..."
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/hairstream-volume-mpl \
    pixi run python scripts/recon_3d/build_volume_partition_bundle.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$BUNDLE_OUT_DIR" \
        --mesh "$MESH_PATH" \
        --views "${VIEWS[@]}" \
        --seed-geometry "$PARTITION_SEED_GEOMETRY"

    if [ -f "${BUNDLE_OUT_DIR}/volume_partition_bundle.npz" ]; then
        VOLUME_PARTITION_BUNDLE="${BUNDLE_OUT_DIR}/volume_partition_bundle.npz"
        echo "  ✓ 体积分区 Bundle 生成完毕: $VOLUME_PARTITION_BUNDLE"
    else
        echo "  [ERROR] build_volume_partition_bundle.py 未能生成 volume_partition_bundle.npz！"
        exit 1
    fi
fi

if run_stage "pde"; then
    echo "=================================================="
    echo " [5/8] PDE: 3D 融合解算 (生成毛发)"
    echo "=================================================="
    PDE_ARGS=(
        "--img_id" "$IMG_ID"
        "--pde_resolution" "384"
        "--pde_dilation_iters" "25"
        "--mesh_obj" "${DATA_DIR}/pixal3d/hair_mesh_aligned_best.obj"
    )

    if [ "$PDE_MODE" = "governed" ]; then
        if [ -z "$VOLUME_PARTITION_BUNDLE" ]; then
            if [ -f "${DATA_DIR}/pde_governance/volume_partition_integration/bundle/volume_partition_bundle.npz" ]; then
                VOLUME_PARTITION_BUNDLE="${DATA_DIR}/pde_governance/volume_partition_integration/bundle/volume_partition_bundle.npz"
            elif [ -f "${DATA_DIR}/pde_governance/volume_partition_integration/volume_partition_bundle.npz" ]; then
                VOLUME_PARTITION_BUNDLE="${DATA_DIR}/pde_governance/volume_partition_integration/volume_partition_bundle.npz"
            else
                FOUND_BUNDLE=($(find "${DATA_DIR}/pde_governance/volume_partition_integration" \( -name "*trusted_partition_bundle*.npz" -o -name "*partition_bundle*.npz" -o -name "*bundle*.npz" \) 2>/dev/null | head -n 1 || true))
                if [ ${#FOUND_BUNDLE[@]} -gt 0 ] && [ -f "${FOUND_BUNDLE[0]}" ]; then
                    VOLUME_PARTITION_BUNDLE="${FOUND_BUNDLE[0]}"
                    echo "  [INFO] 自动采用已审计体积分区 Bundle: $VOLUME_PARTITION_BUNDLE"
                fi
            fi
        fi
        if [ ! -f "$VOLUME_PARTITION_BUNDLE" ]; then
            echo "错误: governed PDE 需要可验证的体积分区 bundle。"
            echo "       未找到: $VOLUME_PARTITION_BUNDLE"
            echo "       请先运行 partition 阶段 (或 build_volume_partition_bundle.py)，或显式使用 --legacy-pde。"
            exit 2
        fi
        GOVERNED_OUT_DIR="${DATA_DIR}/pde_governance/volume_partition_integration/governed_pde"
        PDE_ARGS+=(
            "--pde_solver_mode" "screened_poisson"
            "--volume_partition_bundle" "$VOLUME_PARTITION_BUNDLE"
            "--out_dir" "$GOVERNED_OUT_DIR"
        )
    else
        PDE_ARGS+=("--pde_solver_mode" "legacy_smooth" "--export-per-view")
    fi
    
    # 如果包含侧视角，开启多视角扩展
    OTHER_VIEWS_COUNT=0
    for v in "${VIEWS[@]}"; do
        if [ "$v" != "front" ]; then
            OTHER_VIEWS_COUNT=$((OTHER_VIEWS_COUNT + 1))
        fi
    done

    if [ $OTHER_VIEWS_COUNT -gt 0 ]; then
        PDE_ARGS+=("--views" "${VIEWS[@]}")
    fi

    pixi run python scripts/recon_3d/run_pde_multiview.py "${PDE_ARGS[@]}"
fi

if run_stage "fit_head"; then
    echo "=================================================="
    echo " [6/8] FitHead: 适配 Blender 求值模板头模"
    echo "=================================================="
    # 查找发丝输入: 优先 governed_pde / pde_reconstruction
    INPUT_STRANDS=""
    if [ -f "${DATA_DIR}/pde_governance/volume_partition_integration/governed_pde/final_strands_candidate.npz" ]; then
        INPUT_STRANDS="${DATA_DIR}/pde_governance/volume_partition_integration/governed_pde/final_strands_candidate.npz"
    elif [ -f "${DATA_DIR}/pde_reconstruction/final_strands_candidate.npz" ]; then
        INPUT_STRANDS="${DATA_DIR}/pde_reconstruction/final_strands_candidate.npz"
    elif [ -f "${DATA_DIR}/pde_governance/volume_partition_integration/final_strands_candidate.npz" ]; then
        INPUT_STRANDS="${DATA_DIR}/pde_governance/volume_partition_integration/final_strands_candidate.npz"
    fi

    if [ -z "$INPUT_STRANDS" ]; then
        CANDIDATES=($(find "${DATA_DIR}/pde_governance" \( -name "all_root_prefixes.npz" -o -name "final_strands_candidate.npz" -o -name "*strands*.npz" -o -name "bridge_source.npz" \) 2>/dev/null | head -n 1 || true))
        if [ ${#CANDIDATES[@]} -gt 0 ] && [ -f "${CANDIDATES[0]}" ]; then
            INPUT_STRANDS="${CANDIDATES[0]}"
        fi
    fi

    # 兼容已有 legacy ply 但缺少 npz 的场景
    if [ -z "$INPUT_STRANDS" ] && [ -f "${DATA_DIR}/pde_reconstruction/hair_multiview.ply" ]; then
        echo "  --> 从 ${DATA_DIR}/pde_reconstruction/hair_multiview.ply 转换提取 strands npz..."
        INPUT_STRANDS="${DATA_DIR}/pde_reconstruction/final_strands_candidate.npz"
        pixi run python -c "
import open3d as o3d, numpy as np
ply = o3d.io.read_line_set('${DATA_DIR}/pde_reconstruction/hair_multiview.ply')
pts = np.asarray(ply.points)
lines = np.asarray(ply.lines)
if len(lines) > 0:
    starts = np.where(lines[1:, 0] != lines[:-1, 1])[0] + 1
    strand_indices = np.split(lines, starts)
    strands = []
    for s_lines in strand_indices:
        if len(s_lines) == 0: continue
        strand_pts = [pts[s_lines[0, 0]]] + [pts[l[1]] for l in s_lines]
        strands.append(strand_pts)
    max_len = max(len(s) for s in strands)
    padded = np.zeros((len(strands), max_len, 3), dtype=np.float32)
    for i, s in enumerate(strands):
        padded[i, :len(s)] = s
        padded[i, len(s):] = s[-1]
    np.savez_compressed('${INPUT_STRANDS}', strands=padded)
"
    fi

    if [ -z "$INPUT_STRANDS" ] || [ ! -f "$INPUT_STRANDS" ]; then
        echo "错误: 未找到发丝输入文件 (预期为 ${DATA_DIR}/pde_reconstruction/final_strands_candidate.npz)。请先完成 pde 阶段。"
        exit 1
    fi

    # 检查求值头模是否存在，若不存在则调用 Blender 导出
    HEAD_NPZ="$TEMPLATE_HEAD_PATH"
    if [ -z "$HEAD_NPZ" ]; then
        CANDIDATE_HEAD=($(find "${DATA_DIR}" -name "template_head.npz" 2>/dev/null | head -n 1 || true))
        if [ ${#CANDIDATE_HEAD[@]} -gt 0 ] && [ -f "${CANDIDATE_HEAD[0]}" ]; then
            HEAD_NPZ="${CANDIDATE_HEAD[0]}"
        elif [ -f "assets/template_head.npz" ]; then
            HEAD_NPZ="assets/template_head.npz"
        else
            HEAD_DIR="${DATA_DIR}/template_head"
            echo "  --> 正在从 assets/render_template.blend 导出求值头模到 ${HEAD_DIR}..."
            rm -rf "$HEAD_DIR"
            blender -b assets/render_template.blend -P scripts/render/export_template_head.py -- --output-dir "$HEAD_DIR"
            HEAD_NPZ="${HEAD_DIR}/template_head.npz"
        fi
    fi

    if [ ! -f "$HEAD_NPZ" ]; then
        echo "错误: 无法获取模板头模 $HEAD_NPZ。"
        exit 1
    fi

    FIT_OUT="${DATA_DIR}/template_head_fit"
    if [ -d "$FIT_OUT" ]; then
        echo "  --> 清理旧贴合输出目录: $FIT_OUT"
        rm -rf "$FIT_OUT"
    fi

    echo "  --> 正在执行模板头模贴合与间隙投影..."
    pixi run python scripts/recon_3d/fit_strands_to_template_head.py \
        --input "$INPUT_STRANDS" \
        --head "$HEAD_NPZ" \
        --output-dir "$FIT_OUT"

    echo "  ✓ 模板头模贴合完成: ${FIT_OUT}/all_root_prefixes.npz"
fi

if run_stage "grow_surface"; then
    echo "=================================================="
    echo " [7/8] GrowSurface: 可见表面缺口发丝补生长"
    echo "=================================================="
    # 查找输入发丝：优先 template_head_fit/all_root_prefixes.npz
    GROW_INPUT=""
    if [ -f "${DATA_DIR}/template_head_fit/all_root_prefixes.npz" ]; then
        GROW_INPUT="${DATA_DIR}/template_head_fit/all_root_prefixes.npz"
    elif [ -f "${DATA_DIR}/pde_reconstruction/final_strands_candidate.npz" ]; then
        GROW_INPUT="${DATA_DIR}/pde_reconstruction/final_strands_candidate.npz"
    fi

    if [ -z "$GROW_INPUT" ] || [ ! -f "$GROW_INPUT" ]; then
        echo "错误: grow_surface 需要输入发丝 (请先完成 fit_head 阶段)。"
        exit 1
    fi

    # 头模路径
    HEAD_NPZ="$TEMPLATE_HEAD_PATH"
    if [ -z "$HEAD_NPZ" ]; then
        CANDIDATE_HEAD=($(find "${DATA_DIR}" -name "template_head.npz" 2>/dev/null | head -n 1 || true))
        if [ ${#CANDIDATE_HEAD[@]} -gt 0 ] && [ -f "${CANDIDATE_HEAD[0]}" ]; then
            HEAD_NPZ="${CANDIDATE_HEAD[0]}"
        elif [ -f "assets/template_head.npz" ]; then
            HEAD_NPZ="assets/template_head.npz"
        else
            HEAD_DIR="${DATA_DIR}/template_head"
            rm -rf "$HEAD_DIR"
            blender -b assets/render_template.blend -P scripts/render/export_template_head.py -- --output-dir "$HEAD_DIR"
            HEAD_NPZ="${HEAD_DIR}/template_head.npz"
        fi
    fi

    # 导出或获取模板相机
    CAM_DIR="${DATA_DIR}/template_cameras"
    if [ ! -f "${CAM_DIR}/${SURFACE_GROWTH_VIEW}.npz" ]; then
        echo "  --> 正在导出 Blender 模板评估相机到 ${CAM_DIR}..."
        rm -rf "$CAM_DIR"
        blender -b assets/render_template.blend -P scripts/render/export_template_camera.py -- --output-dir "$CAM_DIR"
    fi

    GROW_OUT="${DATA_DIR}/surface_growth"
    if [ -d "$GROW_OUT" ]; then
        echo "  --> 清理旧补生长输出目录: $GROW_OUT"
        rm -rf "$GROW_OUT"
    fi

    GROW_ARGS=(
        "--data-dir" "$DATA_DIR"
        "--input" "$GROW_INPUT"
        "--head" "$HEAD_NPZ"
        "--output-dir" "$GROW_OUT"
        "--view" "$SURFACE_GROWTH_VIEW"
        "--budget" "$SURFACE_GROWTH_BUDGET"
        "--surface-offset-m" "$SURFACE_OFFSET_M"
    )
    if [ -f "${CAM_DIR}/${SURFACE_GROWTH_VIEW}.npz" ]; then
        GROW_ARGS+=("--coverage-camera" "${CAM_DIR}/${SURFACE_GROWTH_VIEW}.npz")
    fi

    echo "  --> 正在执行表面缺口补生长 (视角: $SURFACE_GROWTH_VIEW, 预算: $SURFACE_GROWTH_BUDGET)..."
    pixi run python scripts/recon_3d/grow_visible_surface_gaps.py "${GROW_ARGS[@]}"

    if [ -f "${GROW_OUT}/all_root_prefixes.npz" ]; then
        pixi run python -c "
import numpy as np
from scripts.recon_3d.recover_boundary_strands import export_all
d = np.load('${GROW_OUT}/all_root_prefixes.npz')
export_all('${GROW_OUT}/all_root_prefixes.ply', d['strands'])
" 2>/dev/null || true
    fi
    echo "  ✓ 表面缺口补生长完成: ${GROW_OUT}/all_root_prefixes.npz"
fi

if run_stage "preview"; then
    echo "=================================================="
    echo " [8/8] Preview: 渲染最终 3D 毛发预览图"
    echo "=================================================="
    PREVIEW_RENDER_NPZ=""
    if [ -f "${DATA_DIR}/surface_growth/all_root_prefixes.npz" ]; then
        PREVIEW_RENDER_NPZ="${DATA_DIR}/surface_growth/all_root_prefixes.npz"
    elif [ -f "${DATA_DIR}/template_head_fit/all_root_prefixes.npz" ]; then
        PREVIEW_RENDER_NPZ="${DATA_DIR}/template_head_fit/all_root_prefixes.npz"
    fi

    if [ -n "$PREVIEW_RENDER_NPZ" ]; then
        PREVIEW_OUT="${DATA_DIR}/preview_renders"
        if [ -d "$PREVIEW_OUT" ]; then
            rm -rf "$PREVIEW_OUT"
        fi
        echo "  --> 使用 Blender 模板渲染绑定的发丝前缀: $PREVIEW_RENDER_NPZ"
        blender -b assets/render_template.blend -P scripts/render/render_all_prefixes.py -- \
            --input "$PREVIEW_RENDER_NPZ" \
            --output-dir "$PREVIEW_OUT" \
            --views "${VIEWS[@]}"
        echo "  ✓ 渲染预览图已保存到: ${PREVIEW_OUT}/"
    else
        # 回退至旧版未绑定发丝预览
        PLY_PATH="${DATA_DIR}/pde_reconstruction/hair_multiview.ply"
        if [ ! -f "$PLY_PATH" ]; then
            echo "警告: 未找到 $PLY_PATH 或有效 template-bound 发丝，跳过预览阶段。"
        else
            blender -b --factory-startup assets/render_template.blend -P scripts/render/render_blender.py -- \
                --hair_path "$PLY_PATH" \
                --save_blend "${DATA_DIR}/pde_reconstruction/scene_with_hair.blend" \
                --output_path "${DATA_DIR}/pde_reconstruction/preview.png"
            echo "  ✓ 旧版 PLY 渲染预览图已保存到: ${DATA_DIR}/pde_reconstruction/preview.png"
        fi
    fi
fi

echo "=================================================="
echo " Pipeline 全部完成！输出目录: $DATA_DIR"
echo "=================================================="
