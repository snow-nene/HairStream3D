# Scripts Directory

This directory contains all the utility, training, inference, rendering, and testing scripts used in the HairStep pipeline. They have been organized into functional subdirectories to make the codebase easier to navigate.

## Structure

- **`train/`**: Scripts for training models.
  - `train_hisa.py`: Main training loop for the strand-map model.
  - `pretrain_structure.py`: Pretraining the model using structure tensor data.
  - `generate_structure_labels.py`: Generate pseudo-labels for pretraining.
  - `batch_generate_masks.py`: Batch process datasets to generate masks.
  - `run_pretrain_full.sh`: Bash script to run the full pretraining pipeline.

- **`test/`**: Scripts for evaluation and overfitting tests.
  - `test_hisa.py`: Test script for the models.
  - `eval_losses.py`: Evaluates and reports losses across splits.
  - `overfit_hrnet_w32_and_test.py`: Overfitting test for HRNet.
  - `overfit_unet_and_test.py`: Overfitting test for U-Net.
  - `test_untrained_unet.py`: Tests untrained networks.

- **`infer_2d/`**: 2D processing pipelines, including image-to-map inferences.
  - `img2hairstep.py`, `img2strand.py`, `img2depth.py`, `img2masks.py`: Various inferences from image to respective 2D maps.
  - `flux_redraw.py`, `flux_redraw_single.py`, `flux_redraw_multiview.py`: Scripts to redraw images using Flux.
  - `standard_pipeline.py`: A standardized 2D inference pipeline.
  - `run_pixal3d.py`: Pixal3D specific scripts.

- **`recon_3d/`**: 3D Generation and PDE simulation scripts.
  - `recon3D.py`, `recon3D_strategy.py`: Logic for 3D reconstruction.
  - `run_pde_generation.py`, `run_pde_multiview.py`: Generates the PDE-based orientation fields.
  - `extract_hair_mesh.py`, `extract_hair_mesh_3d.py`, `extract_hair_mesh_flame.py`: Scripts to extract actual hair meshes from data.

- **`render/`**: Blender and rendering tools.
  - `render_blender.py`, `render_multiview_blender.py`: Wrappers around Blender rendering logic.
  - `render_glb_front.py`, `render_reconstructed.py`: Specific rendering paths and feature map extraction.
  - `compute_multiview_maps.py`: Handles computing multi-view maps.

- **`utils/`**: General utilities and geometry alignments.
  - `align_and_extract_hair.py`, `align_mesh_to_head.py`, `align_glb_lmk.py`: Scripts handling hair and mesh alignment.
  - `find_best_axis.py`: Utilities for finding best alignment axes.
  - `fix_calib_and_proj.py`: Fix calibration/projection matrices.
  - `get_lmk.py`: Extract landmarks.
  - `opt_cam.py`: Optimize camera parameters.

- **`vis/`**: Visualization tools.
  - `vis_hair_direction.py`: Visualizes the extracted hair direction.
  - `vis_laplace_results.py`: Visualizes the Laplace PDE results.
  - `vis_rk4_results.py`: Visualizes the RK4 simulation outputs.
  - `compare_180deg.py`: Diagnostic visualizations.

All imports and references in documentation (such as `docs/`) and internal `from scripts.x import y` have been correctly updated to follow this structure.
