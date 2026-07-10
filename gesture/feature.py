import math
from dataclasses import dataclass
import numpy as np
from gesture.scrolls import Scroll, PathPoint
import doctest

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
    straightness: float # 弦长 / 路径长, ∈ (0, 1], 越接近 1 越直
    direction_change: float # 首段与末段运动向量的夹角, ∈ [0, π]
    '''
    位置特征
    '''
    position_begin_x: float # 初始轨迹点的 X 坐标
    position_begin_y: float # 初始轨迹点的 Y 坐标
    position_end_x: float # 终止轨迹点的 X 坐标
    position_end_y: float # 终止轨迹点的 Y 坐标
    '''
    时间特征
    '''
    duration: float # 手势总时长
    velocity_peak_position: float # 峰速出现的相对位置, ∈ [0, 1]
    '''
    候选相对特征 (默认不在 FEATURE_NAMES, 用作 A/B 备选).
    绝对量的短板是会随用户滑动快慢/屏幕总长整体缩放, 这里预先算好相对形态,
    在 dataset.py 的白名单里加进去即可参与训练.
    '''
    velocity_cv: float # velocity_std / velocity_mean, 变异系数, 说明"匀速 vs 突刺"
    velocity_burst: float # velocity_max / velocity_mean, 峰值/均值, 说明"爆发性"
    disp_ratio_x: float # disp_total_dx / length, 净 X 位移占路径长, ∈ [-1, 1]
    disp_ratio_y: float # disp_total_dy / length, 净 Y 位移占路径长, ∈ [-1, 1]
    bbox_aspect: float # disp_max_dx / (disp_max_dx + disp_max_dy), 包围盒横竖占比 ∈ [0, 1]
    tail_velocity_ratio: float # 末 20% 时间段均速 / 全程均速, <1 减速, >1 加速
    decel_time_ratio: float # (总时长 - 峰速时刻) / 总时长, 减速段占比 ∈ [0, 1]
    ends_near_edge: int # 终点是否落在屏幕 5% 边缘带内 (0/1)


def _length_stats(path: list[PathPoint]) -> float:
    """
    >>> from gesture.scrolls import PathPoint as P
    >>> _length_stats([P(0, 0.0, 0.0), P(1, 3.0, 4.0)])
    5.0
    >>> _length_stats([])
    0.0
    """
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += math.hypot(b.x - a.x, b.y - a.y)
    return total

def _displacement_stats(path: list[PathPoint]) -> tuple[float, ...]:
    """
    >>> from gesture.scrolls import PathPoint as P
    >>> _displacement_stats([P(0, 0.0, 0.0), P(1, 1.0, 2.0), P(2, 3.0, 1.0)])
    (3.0, 1.0, 3.0, 2.0)
    >>> _displacement_stats([P(0, 0.5, 0.5)])
    (0.0, 0.0, 0.0, 0.0)
    """
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
    """
    >>> from gesture.scrolls import PathPoint as P
    >>> velocities([P(0, 0.0, 0.0), P(2, 3.0, 4.0), P(2, 5.0, 5.0)])
    [2.5]
    >>> velocities([P(0, 0.0, 0.0)])
    []
    """
    result: list[float] = []
    for a, b in zip(path, path[1:]):
        dt = b.t - a.t
        if dt <= 0:
            continue
        result.append(math.hypot(b.x - a.x, b.y - a.y) / dt)
    return result


def _velocity_stats(path: list[PathPoint]) -> tuple[float, float, float]:
    """
    >>> from gesture.scrolls import PathPoint as P
    >>> _velocity_stats([P(0, 0.0, 0.0), P(1, 1.0, 0.0), P(3, 3.0, 0.0)])
    (1.0, 1.0, 0.0)
    >>> _velocity_stats([])
    (0.0, 0.0, 0.0)
    """
    vs = velocities(path)
    if not vs:
        return 0.0, 0.0, 0.0
    v_max = max(vs)
    total_duration = path[-1].t - path[0].t
    if total_duration > 0:
        v_mean = _length_stats(path) / total_duration
    else:
        v_mean = sum(vs) / len(vs)
    v_std = math.sqrt(sum((v - v_mean) ** 2 for v in vs) / len(vs))
    return v_max, v_mean, v_std


def _rotate_to_principal_axis(path: list[PathPoint]) -> tuple[np.ndarray, np.ndarray]:
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
    if len(path) < degree + 2:
        return None
    u, v = _rotate_to_principal_axis(path)
    if np.ptp(u) < MIN_FIT_SPAN:
        return None
    poly = np.polynomial.Polynomial.fit(u, v, degree)
    return u, v, poly


def _curvatures_from_fit(u: np.ndarray, poly) -> np.ndarray:
    fp = poly.deriv(1)(u)
    fpp = poly.deriv(2)(u)
    return fpp / np.power(1.0 + fp ** 2, 1.5)


def curvatures(path: list[PathPoint]) -> np.ndarray:
    fit = _fit_polynomial(path)
    if fit is None:
        return np.array([])
    u, _v, poly = fit
    return _curvatures_from_fit(u, poly)


def _shape_stats(
    path: list[PathPoint], eps: float = 1e-6,
) -> tuple[float, float, float, int]:
    """
    >>> from gesture.scrolls import PathPoint as P
    >>> _shape_stats([P(0, 0.0, 0.0), P(1, 0.5, 0.0), P(2, 1.0, 0.0)])
    (0.0, 0.0, 0.0, 0)
    >>> _, _, _, cco = _shape_stats([P(0, 0.0, 0.0), P(1, 0.5, 0.2), P(2, 1.0, 0.0)])
    >>> cco
    -1
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

def _position_stats(path: list[PathPoint]) -> tuple[float, float, float, float]:
    """
    >>> from gesture.scrolls import PathPoint as P
    >>> _position_stats([P(0, 0.1, 0.2), P(1, 0.5, 0.6), P(2, 0.9, 0.4)])
    (0.1, 0.2, 0.9, 0.4)
    >>> _position_stats([])
    (0.0, 0.0, 0.0, 0.0)
    """
    if not path:
        return 0.0, 0.0, 0.0, 0.0
    return path[0].x, path[0].y, path[-1].x, path[-1].y


def _straightness(path: list[PathPoint], length: float, eps: float = 1e-9) -> float:
    """
    >>> from gesture.scrolls import PathPoint as P
    >>> _straightness([P(0, 0.0, 0.0), P(1, 1.0, 0.0), P(2, 2.0, 0.0)], length=2.0)
    1.0
    >>> round(_straightness([P(0, 0.0, 0.0), P(1, 1.0, 1.0), P(2, 2.0, 0.0)], length=math.sqrt(8)), 4)
    0.7071
    >>> _straightness([], length=0.0)
    0.0
    """
    if len(path) < 2 or length <= eps:
        return 0.0
    chord = math.hypot(path[-1].x - path[0].x, path[-1].y - path[0].y)
    return chord / length


def _direction_change(path: list[PathPoint], eps: float = 1e-9) -> float:
    """
    >>> from gesture.scrolls import PathPoint as P
    >>> _direction_change([P(0, 0.0, 0.0), P(1, 1.0, 0.0), P(2, 2.0, 0.0)])
    0.0
    >>> round(_direction_change([P(0, 0.0, 0.0), P(1, 1.0, 0.0), P(2, 1.0, 1.0)]), 4)
    1.5708
    >>> _direction_change([P(0, 0.0, 0.0), P(1, 1.0, 0.0)])
    0.0
    """
    if len(path) < 3:
        return 0.0
    ax, ay = path[1].x - path[0].x, path[1].y - path[0].y
    bx, by = path[-1].x - path[-2].x, path[-1].y - path[-2].y
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na < eps or nb < eps:
        return 0.0
    cos_t = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.acos(cos_t)


def _time_stats(path: list[PathPoint]) -> tuple[float, float]:
    """
    >>> from gesture.scrolls import PathPoint as P
    >>> _time_stats([P(0, 0.0, 0.0), P(1, 1.0, 0.0), P(3, 5.0, 0.0)])
    (3, 1.0)
    >>> _time_stats([P(0, 0.0, 0.0)])
    (0.0, 0.0)
    """
    if len(path) < 2:
        return 0.0, 0.0
    duration = path[-1].t - path[0].t
    vs = velocities(path)
    if not vs:
        return max(duration, 0.0), 0.0
    peak_idx = max(range(len(vs)), key=vs.__getitem__)
    peak_pos = peak_idx / (len(vs) - 1) if len(vs) > 1 else 0.0
    return max(duration, 0.0), peak_pos


_EPS_REL = 1e-9

# 归一化坐标下, 视终点距 [0,1] 边界多近算"划到屏幕边".
_EDGE_MARGIN = 0.05

# tail_velocity_ratio 的"末段"定义: 时间轴最后 20%.
_TAIL_FRAC = 0.20


def _tail_velocity_ratio(path: list[PathPoint]) -> float:
    """Mean speed over the last _TAIL_FRAC of the time axis, / mean speed overall.

    < 1 = decelerating tail (typical for thumbs, gesture ends with a natural stop);
    > 1 = accelerating tail (typical for index-finger flicks that slide off screen).
    """
    if len(path) < 3:
        return 0.0
    vs: list[float] = []
    ts_mid: list[float] = []
    for a, b in zip(path, path[1:]):
        dt = b.t - a.t
        if dt <= 0:
            continue
        vs.append(math.hypot(b.x - a.x, b.y - a.y) / dt)
        ts_mid.append(0.5 * (a.t + b.t))
    if not vs:
        return 0.0
    v_mean = sum(vs) / len(vs)
    if v_mean <= _EPS_REL:
        return 0.0
    span = ts_mid[-1] - ts_mid[0]
    if span <= 0:
        return 1.0
    cutoff = ts_mid[-1] - span * _TAIL_FRAC
    tail = [v for v, t in zip(vs, ts_mid) if t >= cutoff]
    if not tail:
        return 1.0
    return (sum(tail) / len(tail)) / v_mean


def _decel_time_ratio(path: list[PathPoint]) -> float:
    """Fraction of total time spent after the peak-velocity sample.

    Bell-shaped speed profile -> ~0.5 (roughly equal accel/decel).
    Ramp that peaks near the end -> ~0.
    """
    if len(path) < 3:
        return 0.0
    vs: list[float] = []
    ts_mid: list[float] = []
    for a, b in zip(path, path[1:]):
        dt = b.t - a.t
        if dt <= 0:
            continue
        vs.append(math.hypot(b.x - a.x, b.y - a.y) / dt)
        ts_mid.append(0.5 * (a.t + b.t))
    if not vs:
        return 0.0
    span = ts_mid[-1] - ts_mid[0]
    if span <= 0:
        return 0.0
    peak_idx = max(range(len(vs)), key=vs.__getitem__)
    return (ts_mid[-1] - ts_mid[peak_idx]) / span


def _ends_near_edge(path: list[PathPoint]) -> int:
    """1 if the final touch sample is within _EDGE_MARGIN of any screen edge.

    Index-finger flicks frequently slide off the edge; thumbs generally
    stop within the reachable arc. Coordinates are already normalized to
    [0, 1], so the check is a straight compare.
    """
    if not path:
        return 0
    p = path[-1]
    if p.x <= _EDGE_MARGIN or p.x >= 1.0 - _EDGE_MARGIN:
        return 1
    if p.y <= _EDGE_MARGIN or p.y >= 1.0 - _EDGE_MARGIN:
        return 1
    return 0


def compute(scroll: Scroll) -> Features:
    path = scroll.path
    length = _length_stats(path)
    v_max, v_mean, v_std = _velocity_stats(path)
    total_dx, total_dy, max_dx, max_dy = _displacement_stats(path)
    rmse_val, k_max, k_mean, cco = _shape_stats(path)
    straight = _straightness(path, length)
    dir_change = _direction_change(path)
    begin_x, begin_y, end_x, end_y = _position_stats(path)
    duration, v_peak_pos = _time_stats(path)

    v_cv = v_std / v_mean if v_mean > _EPS_REL else 0.0
    v_burst = v_max / v_mean if v_mean > _EPS_REL else 0.0
    disp_x_ratio = total_dx / length if length > _EPS_REL else 0.0
    disp_y_ratio = total_dy / length if length > _EPS_REL else 0.0
    bbox_sum = max_dx + max_dy
    bbox_aspect = max_dx / bbox_sum if bbox_sum > _EPS_REL else 0.0
    tail_v_ratio = _tail_velocity_ratio(path)
    decel_ratio = _decel_time_ratio(path)
    edge_flag = _ends_near_edge(path)

    return Features(
        length=length,
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
        straightness=straight,
        direction_change=dir_change,
        position_begin_x=begin_x,
        position_begin_y=begin_y,
        position_end_x=end_x,
        position_end_y=end_y,
        duration=duration,
        velocity_peak_position=v_peak_pos,
        velocity_cv=v_cv,
        velocity_burst=v_burst,
        disp_ratio_x=disp_x_ratio,
        disp_ratio_y=disp_y_ratio,
        bbox_aspect=bbox_aspect,
        tail_velocity_ratio=tail_v_ratio,
        decel_time_ratio=decel_ratio,
        ends_near_edge=edge_flag,
    )
