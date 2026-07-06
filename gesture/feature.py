import math
from dataclasses import dataclass
import numpy as np
from gesture.scrolls import Scroll, PathPoint

@dataclass
class Features:
    '''
    长度特征
    '''
    length: float # 轨迹总长度
    '''
    位移特征
    '''
    disp_total_dx: float # 轨迹 X 方向总位移
    disp_total_dy: float # 轨迹 Y 方向总位移
    disp_max_dx: float # 轨迹 X 方向最大位移跨度
    disp_max_dy: float # 轨迹 Y 方向最大位移跨度
    '''
    速率特征
    '''
    velocity_max: float # 轨迹最大速率
    velocity_mean: float # 轨迹速率平均值
    velocity_std: float # 轨迹速率的标准差
    '''
    形状特征
    '''
    curvature_rmse: float # 曲率的均方根误差
    curvature_max: float # 曲率的最大值
    curvature_mean: float # 曲率的平均值
    convex_orientation: int # 轨迹凸出的朝向



def _length(path: list[PathPoint]) -> float:
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += math.hypot(b.x - a.x, b.y - a.y)
    return total

def _displacement_stats(path: list[PathPoint]) -> tuple[float, float, float, float]:
    if len(path) < 2:
        return 0.0, 0.0, 0.0, 0.0
    x_first = path[0].x
    y_first = path[0].y
    x_min = x_max = x_first
    y_min = y_max = y_first
    for p in path:
        if p.x < x_min: x_min = p.x
        if p.x > x_max: x_max = p.x
        if p.y < y_min: y_min = p.y
        if p.y > y_max: y_max = p.y
    total_dx = path[-1].x - x_first
    total_dy = path[-1].y - y_first
    return total_dx, total_dy, x_max - x_min, y_max - y_min


def velocities(path: list[PathPoint]) -> list[float]:
    result: list[float] = []
    for a, b in zip(path, path[1:]):
        dt = b.t - a.t
        if dt <= 0:
            continue
        result.append(math.hypot(b.x - a.x, b.y - a.y) / dt)
    return result


def _velocity_stats(path: list[PathPoint]) -> tuple[float, float, float]:
    """Return (v_max, v_mean, v_std).

    v_mean is the time-weighted average = total_length / total_duration,
    matching the spec in tasks.md and avoiding sampling-rate bias. When
    the total duration is zero (bad data), fall back to the arithmetic
    mean of the per-segment velocities.
    """
    vs = velocities(path)
    if not vs:
        return 0.0, 0.0, 0.0
    v_max = max(vs)
    total_duration = path[-1].t - path[0].t
    if total_duration > 0:
        v_mean = _length(path) / total_duration
    else:
        v_mean = sum(vs) / len(vs)
    v_std = math.sqrt(sum((v - v_mean) ** 2 for v in vs) / len(vs))
    return v_max, v_mean, v_std


def _rotate_to_principal_axis(path: list[PathPoint]) -> tuple[np.ndarray, np.ndarray]:
    """Rotate coordinates so first->last vector aligns with +x axis.

    Returns (u, v) where u is the coordinate along the principal axis and v
    is perpendicular. This makes f(u) well-defined for polynomial fitting.
    """
    if len(path) < 2:
        return np.array([p.x for p in path]), np.array([p.y for p in path])
    xs = np.array([p.x for p in path])
    ys = np.array([p.y for p in path])
    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    theta = math.atan2(dy, dx)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    xs0 = xs - xs[0]
    ys0 = ys - ys[0]
    u = cos_t * xs0 + sin_t * ys0
    v = -sin_t * xs0 + cos_t * ys0
    return u, v


MIN_FIT_SPAN = 0.05


def _fit_polynomial(path: list[PathPoint], degree: int = 4):
    """Fit v = p(u) in the rotated frame.

    Returns (u, v, poly) where poly is a numpy.polynomial.Polynomial object,
    or None if the path is too short to fit reliably.
    """
    if len(path) < degree + 2:
        return None
    u, v = _rotate_to_principal_axis(path)
    if np.ptp(u) < MIN_FIT_SPAN:
        return None
    poly = np.polynomial.Polynomial.fit(u, v, degree)
    return u, v, poly


def _curvatures_from_fit(u: np.ndarray, poly) -> np.ndarray:
    """Per-sample signed curvature given an already-fitted polynomial.

    Split out so that `_curvatures` and `_shape_stats` share the derivative
    + curvature formula without either function duplicating it.
    """
    fp = poly.deriv(1)(u)
    fpp = poly.deriv(2)(u)
    return fpp / np.power(1.0 + fp ** 2, 1.5)


def curvatures(path: list[PathPoint]) -> np.ndarray:
    """Per-sample signed curvature. Empty array if the fit is undefined.

    plots.py uses this for per-sample curvature curves. Aggregated shape
    features go through `_shape_stats`, which fits the polynomial only
    once and computes RMSE + curvature stats + CCO together.
    """
    fit = _fit_polynomial(path)
    if fit is None:
        return np.array([])
    u, _v, poly = fit
    return _curvatures_from_fit(u, poly)


def _shape_stats(
    path: list[PathPoint], eps: float = 1e-6,
) -> tuple[float, float, float, int]:
    """Return (rmse, curvature_max, curvature_mean, convex_orientation).

    All four shape features derive from the same rotated-frame polynomial
    fit, so this function fits once and computes them together.

    - rmse: residual std of the poly-fit, i.e. how "jittery" the trajectory
      is relative to a smooth curve.
    - curvature_max / mean: |κ| over the samples.
    - convex_orientation: three-point cross-product sign in {-1, 0, +1},
      normalized by the primary motion axis so the label is stable under
      swipe direction. 0 means the trajectory is too close to a line to
      decide.
    """
    # --- polynomial-fit based features -------------------------------
    fit = _fit_polynomial(path)
    if fit is None:
        rmse_val = k_max = k_mean = 0.0
    else:
        u, v, poly = fit
        residuals = v - poly(u)
        rmse_val = float(np.sqrt(np.mean(residuals ** 2)))
        abs_k = np.abs(_curvatures_from_fit(u, poly))
        k_max = float(abs_k.max())
        k_mean = float(abs_k.mean())

    # --- convex orientation (no fit needed, 3-point sign) ------------
    cco = 0
    if len(path) >= 3:
        p_start, p_end = path[0], path[-1]
        p_mid = path[len(path) // 2]
        v1x, v1y = p_start.x - p_mid.x, p_start.y - p_mid.y
        v2x, v2y = p_end.x - p_mid.x, p_end.y - p_mid.y
        z = v2x * v1y - v2y * v1x
        if abs(z) >= eps:
            dx = p_end.x - p_start.x
            dy = p_end.y - p_start.y
            norm = dy if abs(dy) >= abs(dx) else dx
            if norm != 0:
                cco = int(math.copysign(1, norm * z))

    return rmse_val, k_max, k_mean, cco


def compute(scroll: Scroll) -> Features:
    path = scroll.path
    v_max, v_mean, v_std = _velocity_stats(path)
    total_dx, total_dy, max_dx, max_dy = _displacement_stats(path)
    rmse_val, k_max, k_mean, cco = _shape_stats(path)
    return Features(
        length=_length(path),
        disp_total_dx=total_dx,
        disp_total_dy=total_dy,
        disp_max_dx=max_dx,
        disp_max_dy=max_dy,
        velocity_max=v_max,
        velocity_mean=v_mean,
        velocity_std=v_std,
        curvature_rmse=rmse_val,
        curvature_max=k_max,
        curvature_mean=k_mean,
        convex_orientation=cco,
    )
