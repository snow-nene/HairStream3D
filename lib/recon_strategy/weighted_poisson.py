"""Masked weighted screened-Poisson solver for 3D orientation fields.

The solver minimizes a discrete form of

    integral a |grad u|^2 + w |u - d|^2

inside an arbitrary voxel domain. Trusted observations are imposed as strong
Dirichlet conditions; synthesized observations enter through ``soft_weight``.
Domain boundaries use a zero-flux Neumann condition. The implementation is
matrix-free and uses preconditioned conjugate gradients on CPU or CUDA.
"""

from dataclasses import asdict, dataclass
import time
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, label


@dataclass(frozen=True)
class ScreenedPoissonMetrics:
    """Numerical evidence emitted for every PDE solve."""

    solver: str
    resolution: list[int]
    domain_voxels: int
    unknown_voxels: int
    dirichlet_voxels: int
    soft_voxels: int
    iterations: int
    initial_relative_residual: float
    final_relative_residual: float
    converged: bool
    dirichlet_max_error: float
    elapsed_seconds: float
    device: str
    dtype: str
    breakdown: Optional[str] = None
    component_residuals: Optional[list[dict]] = None

    def to_dict(self) -> dict:
        """Return a JSON-serializable metrics dictionary."""

        return asdict(self)


def build_signed_domain_distance(
    domain_mask: np.ndarray,
    b_min: Sequence[float],
    b_max: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a metric signed distance and inward normals for a PDE domain.

    Positive values are inside the domain, negative values are outside, and
    the gradient therefore points inward around both sides of the boundary.
    """

    domain = np.asarray(domain_mask, dtype=bool)
    if domain.ndim != 3 or not domain.any():
        raise ValueError("domain_mask must be a non-empty 3D mask")
    b_min_np = np.asarray(b_min, dtype=np.float64)
    b_max_np = np.asarray(b_max, dtype=np.float64)
    if b_min_np.shape != (3,) or b_max_np.shape != (3,) \
            or np.any(b_max_np <= b_min_np):
        raise ValueError("b_min and b_max must define a valid 3D box")
    spacing = (b_max_np - b_min_np) / np.maximum(
        np.asarray(domain.shape, dtype=np.float64) - 1.0,
        1.0,
    )
    inside = distance_transform_edt(domain, sampling=spacing)
    outside = distance_transform_edt(~domain, sampling=spacing)
    signed = inside - outside
    gradients = np.stack(
        np.gradient(signed, *spacing, edge_order=1), axis=0
    )
    gradient_norm = np.linalg.norm(gradients, axis=0, keepdims=True)
    normals = np.divide(
        gradients,
        np.maximum(gradient_norm, 1e-12),
        out=np.zeros_like(gradients),
    )
    return (
        signed.astype(np.float32),
        normals.astype(np.float32),
        spacing.astype(np.float32),
    )


def limit_outward_domain_direction(
    direction: torch.Tensor,
    signed_distance: torch.Tensor,
    inward_normal: torch.Tensor,
    margin: float,
    inward_bias: float = 0.05,
) -> torch.Tensor:
    """Turn outward directions into a metric-domain tangent/inward direction.

    Merely deleting the outward component can collapse an almost-normal vector
    to zero and freeze RK4 integration on the domain shell. A small inward
    barrier avoids that degeneracy, while normalization preserves speed.
    """

    if margin <= 0.0:
        return direction
    if inward_bias < 0.0:
        raise ValueError("inward_bias must be non-negative")
    normal = torch.nn.functional.normalize(inward_normal, dim=0)
    inward_component = torch.sum(direction * normal, dim=0)
    constrain = (signed_distance <= float(margin)) & (inward_component < 0.0)
    tangent = direction - inward_component.unsqueeze(0) * normal
    direction_norm = torch.norm(direction, dim=0)
    layer_strength = torch.clamp(
        1.0 - signed_distance / float(margin), min=0.05, max=1.0
    )
    barrier = (
        direction_norm * float(inward_bias) * layer_strength
    ).unsqueeze(0) * normal
    corrected = tangent + barrier
    corrected_norm = torch.norm(corrected, dim=0)
    corrected = corrected * (
        direction_norm / torch.clamp(corrected_norm, min=1e-8)
    ).unsqueeze(0)
    return torch.where(constrain.unsqueeze(0), corrected, direction)


def recover_points_inside_domain(
    points: torch.Tensor,
    signed_distance: torch.Tensor,
    inward_normal: torch.Tensor,
    margin: float,
    epsilon: float = 0.001,
) -> torch.Tensor:
    """Project near-boundary query points into a local PDE continuation band."""

    if margin < 0.0 or epsilon < 0.0:
        raise ValueError("margin and epsilon must be non-negative")
    normal = torch.nn.functional.normalize(inward_normal, dim=0)
    correction = torch.clamp(
        float(margin) - signed_distance + float(epsilon),
        min=0.0,
    )
    return points + correction.unsqueeze(0) * normal


def _as_tensor(value, *, device: torch.device, dtype=None) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _validate_anchored_components(
    domain: np.ndarray,
    dirichlet: np.ndarray,
    soft_weight: np.ndarray,
    partition_labels: Optional[np.ndarray] = None,
) -> None:
    """Reject disconnected components without any PDE anchor."""

    components, count = label_partition_components(domain, partition_labels)
    if count == 0:
        raise ValueError("PDE domain is empty")
    anchored = dirichlet | (soft_weight > 0.0)
    anchored_ids = np.unique(components[anchored])
    anchored_ids = anchored_ids[anchored_ids > 0]
    if len(anchored_ids) == count:
        return
    missing = np.setdiff1d(
        np.arange(1, count + 1, dtype=components.dtype),
        anchored_ids,
        assume_unique=True,
    )
    sizes = np.bincount(components.ravel())
    missing_sizes = [int(sizes[index]) for index in missing[:8]]
    raise ValueError(
        "PDE domain contains unanchored connected components: "
        f"count={len(missing)}, example_sizes={missing_sizes}"
    )


def label_partition_components(
    domain: np.ndarray,
    partition_labels: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, int]:
    """Label spatial components while treating partition changes as cuts."""

    domain = np.asarray(domain, dtype=bool)
    if partition_labels is None:
        return label(domain)
    partitions = np.asarray(partition_labels)
    if partitions.shape != domain.shape:
        raise ValueError("partition_labels must match domain_mask")
    if np.any(domain & (partitions <= 0)):
        raise ValueError("partition_labels must be positive inside domain_mask")

    components = np.zeros(domain.shape, dtype=np.int32)
    component_count = 0
    for partition_id in np.unique(partitions[domain]):
        local, local_count = label(domain & (partitions == partition_id))
        active = local > 0
        components[active] = local[active] + component_count
        component_count += int(local_count)
    return components, component_count


def solve_weighted_screened_poisson(
    domain_mask: np.ndarray,
    dirichlet_values: np.ndarray,
    dirichlet_mask: np.ndarray,
    *,
    soft_values: Optional[np.ndarray] = None,
    soft_weight: Optional[np.ndarray] = None,
    normal_penalty_normals: Optional[np.ndarray] = None,
    normal_penalty_weight: Optional[np.ndarray] = None,
    partition_labels: Optional[np.ndarray] = None,
    require_component_convergence: bool = False,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    diffusion: float = 1.0,
    tolerance: float = 1e-4,
    max_iterations: int = 2000,
    initial_field: Optional[np.ndarray] = None,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
    validate_components: bool = True,
) -> tuple[np.ndarray, ScreenedPoissonMetrics]:
    """Solve a masked vector-valued weighted screened-Poisson equation.

    Args:
        domain_mask: Boolean ``(X, Y, Z)`` PDE domain.
        dirichlet_values: Trusted vector values shaped ``(3, X, Y, Z)``.
        dirichlet_mask: Boolean strong-boundary mask.
        soft_values: Optional synthesized observations with the same vector shape.
        soft_weight: Non-negative scalar confidence field shaped ``(X, Y, Z)``.
        normal_penalty_normals: Optional unit normals shaped ``(3, X, Y, Z)``.
        normal_penalty_weight: Optional non-negative weights for the energy
            ``weight * (normal dot field)^2``.
        partition_labels: Optional positive integer labels on the PDE domain.
            Neighbor pairs with different labels have zero diffusion coupling.
        spacing: Physical voxel spacing for X, Y and Z.
        diffusion: Positive spatial smoothness coefficient.
        tolerance: Relative residual convergence threshold.
        max_iterations: Maximum preconditioned-CG iterations.
        initial_field: Optional full-field warm start. EDT is allowed here only.
        device: Torch device. Defaults to CUDA when available, otherwise CPU.
        dtype: Numerical dtype used by the solver.
        validate_components: Check that each domain component has an anchor.

    Returns:
        A ``(field, metrics)`` tuple. The field is not normalized after the
        linear solve, so residual evidence remains tied to the returned PDE
        solution. Callers may normalize a copy for RK4 integration.
    """

    start_time = time.perf_counter()
    domain_np = np.asarray(domain_mask, dtype=bool)
    boundary_np = np.asarray(dirichlet_mask, dtype=bool)
    values_np = np.asarray(dirichlet_values)
    if domain_np.ndim != 3:
        raise ValueError("domain_mask must have shape (X, Y, Z)")
    if boundary_np.shape != domain_np.shape:
        raise ValueError("dirichlet_mask must match domain_mask")
    if values_np.shape != (3, *domain_np.shape):
        raise ValueError("dirichlet_values must have shape (3, X, Y, Z)")
    if np.any(boundary_np & ~domain_np):
        raise ValueError("Dirichlet voxels must be inside the PDE domain")
    if not boundary_np.any():
        raise ValueError("At least one Dirichlet voxel is required")
    partition_np = None
    if partition_labels is not None:
        partition_np = np.asarray(partition_labels)
        if partition_np.shape != domain_np.shape:
            raise ValueError("partition_labels must match domain_mask")
        if not np.all(np.isfinite(partition_np)):
            raise ValueError("partition_labels must be finite")
        if np.any(domain_np & (partition_np <= 0)):
            raise ValueError(
                "partition_labels must be positive inside domain_mask"
            )
    if diffusion <= 0.0:
        raise ValueError("diffusion must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")

    spacing_np = np.asarray(spacing, dtype=np.float64)
    if spacing_np.shape != (3,) or not np.all(np.isfinite(spacing_np)) \
            or np.any(spacing_np <= 0.0):
        raise ValueError("spacing must contain three positive finite values")

    if soft_weight is None:
        soft_weight_np = np.zeros_like(domain_np, dtype=np.float32)
    else:
        soft_weight_np = np.asarray(soft_weight, dtype=np.float32)
        if soft_weight_np.shape != domain_np.shape:
            raise ValueError("soft_weight must match domain_mask")
        if not np.all(np.isfinite(soft_weight_np)) \
                or np.any(soft_weight_np < 0.0):
            raise ValueError("soft_weight must be finite and non-negative")
        soft_weight_np = np.where(domain_np, soft_weight_np, 0.0)

    if soft_values is None:
        soft_values_np = np.zeros_like(values_np, dtype=np.float32)
    else:
        soft_values_np = np.asarray(soft_values)
        if soft_values_np.shape != values_np.shape:
            raise ValueError("soft_values must match dirichlet_values")
        if not np.all(np.isfinite(soft_values_np)):
            raise ValueError("soft_values must be finite")

    if normal_penalty_weight is None:
        normal_weight_np = np.zeros_like(domain_np, dtype=np.float32)
        normal_values_np = np.zeros_like(values_np, dtype=np.float32)
    else:
        normal_weight_np = np.asarray(normal_penalty_weight, dtype=np.float32)
        normal_values_np = np.asarray(normal_penalty_normals, dtype=np.float32)
        if normal_weight_np.shape != domain_np.shape:
            raise ValueError("normal_penalty_weight must match domain_mask")
        if normal_values_np.shape != values_np.shape:
            raise ValueError("normal_penalty_normals must match dirichlet_values")
        if not np.all(np.isfinite(normal_weight_np)) \
                or np.any(normal_weight_np < 0.0):
            raise ValueError("normal_penalty_weight must be finite and non-negative")
        if not np.all(np.isfinite(normal_values_np)):
            raise ValueError("normal_penalty_normals must be finite")
        normal_weight_np = np.where(domain_np, normal_weight_np, 0.0)

    if validate_components:
        _validate_anchored_components(
            domain_np,
            boundary_np,
            soft_weight_np,
            partition_np,
        )

    solve_device = torch.device(
        device if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    domain = _as_tensor(domain_np, device=solve_device, dtype=torch.bool)
    partitions = (
        _as_tensor(partition_np, device=solve_device, dtype=torch.int64)
        if partition_np is not None else None
    )
    boundary = _as_tensor(boundary_np, device=solve_device, dtype=torch.bool)
    unknown = domain & ~boundary
    boundary_values = _as_tensor(values_np, device=solve_device, dtype=dtype)
    boundary_values = boundary_values * boundary.unsqueeze(0)
    soft = _as_tensor(soft_weight_np, device=solve_device, dtype=dtype)
    target = _as_tensor(soft_values_np, device=solve_device, dtype=dtype)
    normal_weight = _as_tensor(
        normal_weight_np, device=solve_device, dtype=dtype
    )
    penalty_normal = _as_tensor(
        normal_values_np, device=solve_device, dtype=dtype
    )
    penalty_normal = torch.nn.functional.normalize(
        penalty_normal, dim=0, eps=1e-12
    )
    axis_coefficients = [
        float(diffusion / (axis_spacing * axis_spacing))
        for axis_spacing in spacing_np
    ]

    diagonal = soft.clone()
    for axis, coefficient in enumerate(axis_coefficients):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        lower = tuple(lower)
        upper = tuple(upper)
        pair = domain[lower] & domain[upper]
        if partitions is not None:
            pair &= partitions[lower] == partitions[upper]
        diagonal[lower] += coefficient * pair
        diagonal[upper] += coefficient * pair
    component_diagonal = diagonal.unsqueeze(0) + (
        normal_weight.unsqueeze(0) * penalty_normal.square()
    )
    if unknown.any() and torch.any(component_diagonal[:, unknown] <= 0.0):
        raise ValueError("PDE operator has a non-positive diagonal")

    def apply_operator(vector: torch.Tensor) -> torch.Tensor:
        result = soft.unsqueeze(0) * vector
        normal_component = torch.sum(vector * penalty_normal, dim=0)
        result += (
            normal_weight * normal_component
        ).unsqueeze(0) * penalty_normal
        for axis, coefficient in enumerate(axis_coefficients):
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis] = slice(0, -1)
            upper[axis] = slice(1, None)
            lower = tuple(lower)
            upper = tuple(upper)
            pair = domain[lower] & domain[upper]
            if partitions is not None:
                pair &= partitions[lower] == partitions[upper]
            difference = (vector[(slice(None), *lower)]
                          - vector[(slice(None), *upper)])
            flux = coefficient * difference * pair.unsqueeze(0)
            result[(slice(None), *lower)] += flux
            result[(slice(None), *upper)] -= flux
        result *= unknown.unsqueeze(0)
        return result

    rhs = soft.unsqueeze(0) * target - apply_operator(boundary_values)
    rhs *= unknown.unsqueeze(0)
    if initial_field is None:
        solution = torch.zeros_like(boundary_values)
    else:
        initial_np = np.asarray(initial_field)
        if initial_np.shape != values_np.shape:
            raise ValueError("initial_field must match dirichlet_values")
        solution = _as_tensor(initial_np, device=solve_device, dtype=dtype)
        solution = (solution - boundary_values) * unknown.unsqueeze(0)

    rhs_norm = torch.linalg.vector_norm(rhs)
    denominator = torch.clamp(rhs_norm, min=torch.finfo(dtype).eps)
    residual = rhs - apply_operator(solution)
    initial_relative = float(
        (torch.linalg.vector_norm(residual) / denominator).item()
    )
    relative = initial_relative
    converged = relative <= tolerance
    breakdown = None
    iterations = 0

    inverse_diagonal = torch.zeros_like(component_diagonal)
    inverse_diagonal[:, unknown] = 1.0 / component_diagonal[:, unknown]
    preconditioned = residual * inverse_diagonal
    direction = preconditioned.clone()
    rho = torch.sum(residual * preconditioned)

    for iteration in range(1, max_iterations + 1):
        if converged:
            break
        operator_direction = apply_operator(direction)
        direction_operator = torch.sum(direction * operator_direction)
        if not torch.isfinite(direction_operator) or direction_operator <= 0.0:
            breakdown = "non_positive_curvature"
            break
        alpha = rho / direction_operator
        solution += alpha * direction
        residual -= alpha * operator_direction
        iterations = iteration
        relative = float(
            (torch.linalg.vector_norm(residual) / denominator).item()
        )
        if not np.isfinite(relative):
            breakdown = "non_finite_residual"
            break
        if relative <= tolerance:
            converged = True
            break
        preconditioned = residual * inverse_diagonal
        next_rho = torch.sum(residual * preconditioned)
        if not torch.isfinite(next_rho) or torch.abs(rho) \
                <= torch.finfo(dtype).eps:
            breakdown = "invalid_preconditioned_residual"
            break
        direction = preconditioned + (next_rho / rho) * direction
        rho = next_rho

    # Recompute the true residual from A and the final iterate. Recursive CG
    # residuals can drift in finite precision and are not sufficient evidence
    # for the governance convergence gate.
    residual = rhs - apply_operator(solution)
    relative = float(
        (torch.linalg.vector_norm(residual) / denominator).item()
    )
    converged = relative <= tolerance
    if converged:
        breakdown = None

    component_metrics = None
    if require_component_convergence:
        component_labels, component_count = label_partition_components(domain_np, partition_np)
        component_metrics = []
        residual_np = residual.detach().cpu().numpy()
        rhs_np = rhs.detach().cpu().numpy()
        for component_id in range(1, component_count + 1):
            selected = (component_labels == component_id) & ~boundary_np
            rhs_length = float(np.linalg.norm(rhs_np[:, selected].astype(np.float64)))
            residual_length = float(np.linalg.norm(residual_np[:, selected].astype(np.float64)))
            relative_component = residual_length / rhs_length if rhs_length > 0 else None
            absolute_tolerance = float(tolerance)
            passed = (relative_component <= tolerance if rhs_length > 0
                      else residual_length <= absolute_tolerance)
            component_metrics.append({
                "component_id": component_id, "unknown_voxels": int(selected.sum()),
                "rhs_norm": rhs_length, "absolute_residual": residual_length,
                "relative_residual": relative_component,
                "zero_rhs_absolute_tolerance": absolute_tolerance, "converged": bool(passed),
            })
        converged = converged and all(item["converged"] for item in component_metrics)
        if not converged and breakdown is None:
            breakdown = "component_residual_gate_failed"

    field = solution + boundary_values
    field *= domain.unsqueeze(0)
    if boundary.any():
        boundary_error = torch.max(
            torch.abs(field[:, boundary] - boundary_values[:, boundary])
        ).item()
    else:
        boundary_error = 0.0
    field_np = field.detach().cpu().numpy()
    metrics = ScreenedPoissonMetrics(
        solver="weighted_screened_poisson_pcg",
        resolution=[int(value) for value in domain_np.shape],
        domain_voxels=int(domain_np.sum()),
        unknown_voxels=int((domain_np & ~boundary_np).sum()),
        dirichlet_voxels=int(boundary_np.sum()),
        soft_voxels=int((soft_weight_np > 0.0).sum()),
        iterations=iterations,
        initial_relative_residual=initial_relative,
        final_relative_residual=relative,
        converged=converged,
        dirichlet_max_error=float(boundary_error),
        elapsed_seconds=float(time.perf_counter() - start_time),
        device=str(solve_device),
        dtype=str(dtype).replace("torch.", ""),
        breakdown=breakdown,
        component_residuals=component_metrics,
    )
    return field_np, metrics
