"""AFC-SR1 S0 reader for one complete generated TILED cell.

This module deliberately treats the decoded TILED raster as its only
perspective authority.  It finds local contrast-defined quadrilateral cells;
it does not infer room extent, select global line families, or transfer
evidence from another image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Sequence

import cv2
import numpy as np


READER_VERSION: Final = "afc-sr1-tiled-perspective-reader/s0"

# The constants are geometric/numerical only.  Canny operates on grayscale
# contrast, so neither grout colour nor grout light/dark polarity is assumed.
_CANNY_LOW: Final = 40
_CANNY_HIGH: Final = 100
_CONTOUR_APPROXIMATION_RATIO: Final = 0.02
_MIN_AREA_FRACTION: Final = 1e-4
_BORDER_MARGIN_FRACTION: Final = 0.01
_MIN_BORDER_MARGIN_PX: Final = 2.0
_NEAR_FIELD_MIN_Y_FRACTION: Final = 0.5
_GEOMETRY_EPSILON: Final = 1e-8
_HOMOGRAPHY_EPSILON: Final = 1e-12
_REPROJECTION_TOLERANCE_PX: Final = 1e-5

FailureReason = Literal[
    "invalid_input_image",
    "no_complete_tile_found",
    "invalid_tile_geometry",
    "semantic_corner_ordering_failure",
    "homography_failure",
]
Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]
Matrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Vector3 = tuple[float, float, float]


class TiledPerspectiveReaderFailure(ValueError):
    """A small explicit failure surface for the S0 direct Python API."""

    def __init__(self, reason: FailureReason):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class TileCell:
    """One observed, complete cell in decoded TILED pixel coordinates.

    ``pixel_corners`` and ``source_normalized_corners`` are always ordered
    ``NL, NR, FR, FL``.
    """

    pixel_corners: Quad
    source_normalized_corners: Quad


@dataclass(frozen=True)
class TiledPerspectiveSingleTileResult:
    version: str
    decoded_width: int
    decoded_height: int
    selected_tile: TileCell
    h_lattice_to_tiled: Matrix3x3
    vp_a_homogeneous: Vector3
    vp_b_homogeneous: Vector3
    authoritative_quad_pixel: Quad
    authoritative_quad_source_normalized: Quad
    candidate_count: int
    accepted_complete_tile_count: int


def read_tiled_perspective_single_tile(encoded_image: bytes) -> TiledPerspectiveSingleTileResult:
    """Decode immutable TILED image bytes and return one authoritative cell."""
    if not encoded_image:
        raise TiledPerspectiveReaderFailure("invalid_input_image")
    encoded = np.frombuffer(encoded_image, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise TiledPerspectiveReaderFailure("invalid_input_image")
    return read_tiled_perspective_single_tile_image(image)


def read_tiled_perspective_single_tile_image(image: np.ndarray) -> TiledPerspectiveSingleTileResult:
    """Read one complete TILED cell from an already-decoded BGR image.

    Canny contours provide a local, contrast-only four-edge observation.  The
    S0 fixture contains high-contrast non-tile quads (window frames), so the
    candidate lane is limited to cells whose four observed corners are in the
    image's near field (lower half).  This is S0 fixture-lane selection, not
    a general TileCell contract; it does not inspect materials, use wall/floor
    boundaries, or assume grout polarity.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise TiledPerspectiveReaderFailure("invalid_input_image")
    height, width = image.shape[:2]
    if width < 2 or height < 2:
        raise TiledPerspectiveReaderFailure("invalid_input_image")

    candidates = _extract_complete_quad_candidates(image)
    near_field_candidates = [
        candidate
        for candidate in candidates
        if min(float(point[1]) for point in candidate) >= height * _NEAR_FIELD_MIN_Y_FRACTION
    ]
    if not near_field_candidates:
        raise TiledPerspectiveReaderFailure("no_complete_tile_found")

    selected = min(
        near_field_candidates,
        key=lambda candidate: _selection_key(candidate, width=width, height=height),
    )
    return _result_from_ordered_pixel_corners(
        selected,
        width=width,
        height=height,
        candidate_count=len(candidates),
        accepted_complete_tile_count=len(near_field_candidates),
    )


def _extract_complete_quad_candidates(image: np.ndarray) -> list[np.ndarray]:
    """Return local four-edge contour cells satisfying the S0 tile contract."""
    height, width = image.shape[:2]
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    edges = cv2.Canny(blurred, _CANNY_LOW, _CANNY_HIGH)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[np.ndarray] = []
    minimum_area = max(16.0, width * height * _MIN_AREA_FRACTION)
    border_margin = max(_MIN_BORDER_MARGIN_PX, min(width, height) * _BORDER_MARGIN_FRACTION)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if not np.isfinite(perimeter) or perimeter <= _GEOMETRY_EPSILON:
            continue
        approximate = cv2.approxPolyDP(
            contour,
            _CONTOUR_APPROXIMATION_RATIO * perimeter,
            True,
        ).reshape(-1, 2).astype(np.float64)
        if len(approximate) != 4:
            continue
        try:
            ordered = _order_nl_nr_fr_fl(approximate, minimum_area)
        except TiledPerspectiveReaderFailure:
            continue
        if any(
            point[0] <= border_margin
            or point[0] >= width - border_margin
            or point[1] <= border_margin
            or point[1] >= height - border_margin
            for point in ordered
        ):
            continue
        candidates.append(ordered)
    return candidates


def _selection_key(
    corners: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, tuple[float, ...]]:
    """Prefer interior cells, then larger observed area, with stable ties."""
    area = _polygon_area(corners)
    centroid = np.mean(corners, axis=0)
    border_clearance = min(
        float(np.min(corners[:, 0])),
        float(np.min(corners[:, 1])),
        float(width - np.max(corners[:, 0])),
        float(height - np.max(corners[:, 1])),
    )
    return (
        -border_clearance,
        -area,
        float(centroid[1]),
        tuple(float(value) for value in corners.reshape(-1)),
    )


def _order_nl_nr_fr_fl(points: np.ndarray, minimum_area: float) -> np.ndarray:
    """Validate a quadrilateral and assign the AFC ``NL, NR, FR, FL`` order."""
    corners = np.asarray(points, dtype=np.float64)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise TiledPerspectiveReaderFailure("invalid_tile_geometry")
    for index in range(4):
        for other_index in range(index):
            if np.linalg.norm(corners[index] - corners[other_index]) <= _GEOMETRY_EPSILON:
                raise TiledPerspectiveReaderFailure("invalid_tile_geometry")

    centroid = np.mean(corners, axis=0)
    cyclic = corners[np.argsort(np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0]))]
    if not _is_strictly_convex(cyclic, minimum_area):
        raise TiledPerspectiveReaderFailure("invalid_tile_geometry")

    # The two possible opposite-edge pairings are tested.  The usable pairing
    # has a complete lower (near) edge strictly below its complete upper (far)
    # edge, rather than merely comparing individual corner y values.
    pairings: list[tuple[float, np.ndarray, np.ndarray]] = []
    for far_indices, near_indices in (((0, 1), (2, 3)), ((1, 2), (3, 0))):
        far_edge = cyclic[list(far_indices)]
        near_edge = cyclic[list(near_indices)]
        vertical_gap = float(np.min(near_edge[:, 1]) - np.max(far_edge[:, 1]))
        if vertical_gap > _GEOMETRY_EPSILON:
            pairings.append((vertical_gap, far_edge, near_edge))
        reverse_gap = float(np.min(far_edge[:, 1]) - np.max(near_edge[:, 1]))
        if reverse_gap > _GEOMETRY_EPSILON:
            pairings.append((reverse_gap, near_edge, far_edge))
    if not pairings:
        raise TiledPerspectiveReaderFailure("semantic_corner_ordering_failure")

    _, far_edge, near_edge = max(pairings, key=lambda item: item[0])
    far_left, far_right = sorted(far_edge, key=lambda point: (float(point[0]), float(point[1])))
    near_left, near_right = sorted(near_edge, key=lambda point: (float(point[0]), float(point[1])))
    ordered = np.asarray((near_left, near_right, far_right, far_left), dtype=np.float64)
    if not _is_strictly_convex(ordered, minimum_area):
        raise TiledPerspectiveReaderFailure("semantic_corner_ordering_failure")
    return ordered


def _is_strictly_convex(corners: np.ndarray, minimum_area: float) -> bool:
    if _polygon_area(corners) <= minimum_area:
        return False
    cross_products = []
    for index in range(4):
        first = corners[(index + 1) % 4] - corners[index]
        second = corners[(index + 2) % 4] - corners[(index + 1) % 4]
        cross_products.append(_cross_z(first, second))
    return (
        all(value > _GEOMETRY_EPSILON for value in cross_products)
        or all(value < -_GEOMETRY_EPSILON for value in cross_products)
    )


def _polygon_area(corners: np.ndarray) -> float:
    return abs(
        0.5
        * float(
            np.dot(corners[:, 0], np.roll(corners[:, 1], -1))
            - np.dot(corners[:, 1], np.roll(corners[:, 0], -1))
        )
    )


def _cross_z(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _result_from_ordered_pixel_corners(
    ordered_corners: np.ndarray,
    *,
    width: int,
    height: int,
    candidate_count: int,
    accepted_complete_tile_count: int,
) -> TiledPerspectiveSingleTileResult:
    """Fit one unit-square DLT and derive its authoritative projective data."""
    corners = np.asarray(ordered_corners, dtype=np.float64)
    minimum_area = max(16.0, width * height * _MIN_AREA_FRACTION)
    if not _is_strictly_convex(corners, minimum_area):
        raise TiledPerspectiveReaderFailure("invalid_tile_geometry")

    lattice_corners = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64)
    homography, _ = cv2.findHomography(lattice_corners, corners, method=0)
    if homography is None or homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
        raise TiledPerspectiveReaderFailure("homography_failure")
    homography = np.asarray(homography, dtype=np.float64)
    scale = max(1.0, float(np.linalg.norm(homography, ord="fro")))
    if (
        np.linalg.matrix_rank(homography) != 3
        or abs(float(np.linalg.det(homography))) <= _HOMOGRAPHY_EPSILON * scale**3
    ):
        raise TiledPerspectiveReaderFailure("homography_failure")

    authoritative = _map_lattice_points(homography, lattice_corners)
    if authoritative is None or not np.allclose(
        authoritative,
        corners,
        rtol=0.0,
        atol=_REPROJECTION_TOLERANCE_PX,
    ):
        raise TiledPerspectiveReaderFailure("homography_failure")

    vp_a = homography @ np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    vp_b = homography @ np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    if not np.all(np.isfinite(vp_a)) or not np.all(np.isfinite(vp_b)):
        raise TiledPerspectiveReaderFailure("homography_failure")

    pixel_quad = _as_quad(authoritative)
    normalized_quad = _as_quad(
        np.column_stack((authoritative[:, 0] / width, authoritative[:, 1] / height))
    )
    return TiledPerspectiveSingleTileResult(
        version=READER_VERSION,
        decoded_width=width,
        decoded_height=height,
        selected_tile=TileCell(
            pixel_corners=pixel_quad,
            source_normalized_corners=normalized_quad,
        ),
        h_lattice_to_tiled=_as_matrix3x3(homography),
        vp_a_homogeneous=_as_vector3(vp_a),
        vp_b_homogeneous=_as_vector3(vp_b),
        authoritative_quad_pixel=pixel_quad,
        authoritative_quad_source_normalized=normalized_quad,
        candidate_count=candidate_count,
        accepted_complete_tile_count=accepted_complete_tile_count,
    )


def _map_lattice_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    mapped = (homography @ homogeneous.T).T
    if np.any(np.abs(mapped[:, 2]) <= _HOMOGRAPHY_EPSILON):
        return None
    return mapped[:, :2] / mapped[:, 2:3]


def _as_quad(points: np.ndarray) -> Quad:
    return tuple((float(point[0]), float(point[1])) for point in points)  # type: ignore[return-value]


def _as_matrix3x3(matrix: np.ndarray) -> Matrix3x3:
    return tuple(tuple(float(value) for value in row) for row in matrix)  # type: ignore[return-value]


def _as_vector3(vector: np.ndarray) -> Vector3:
    return tuple(float(value) for value in vector)  # type: ignore[return-value]


# S1 keeps its raster tuning separate from S0.  These values only make local
# contour observations; TileCell acceptance and lattice topology below do not
# use S0's lower-half, interior-winner, or semantic-y-overlap heuristics.
_S1_CANNY_LOW: Final = 20
_S1_CANNY_HIGH: Final = 60
# C-T1's matched inner/outer grout contours place shared endpoints up to six
# pixels apart; eight pixels joins that raster offset while remaining far below
# the shortest trusted tile edge.
_S1_VERTEX_SNAP_TOLERANCE_PX: Final = 8.0
_S1_DUPLICATE_VERTEX_TOLERANCE_PX: Final = 3.0
_EDGE_CORNERS: Final = ((0, 1), (1, 2), (2, 3), (3, 0))
_ORIENTATIONS: Final = (
    ((0, 0), (1, 0), (1, 1), (0, 1)),
    ((1, 0), (1, 1), (0, 1), (0, 0)),
    ((1, 1), (0, 1), (0, 0), (1, 0)),
    ((0, 1), (0, 0), (1, 0), (1, 1)),
    ((1, 0), (0, 0), (0, 1), (1, 1)),
    ((1, 1), (1, 0), (0, 0), (0, 1)),
    ((0, 1), (1, 1), (1, 0), (0, 0)),
    ((0, 0), (0, 1), (1, 1), (1, 0)),
)

S1FailureReason = Literal[
    "invalid_input_image",
    "no_complete_tile",
    "no_coherent_lattice",
    "lattice_assignment_conflict",
    "no_rectangular_core",
    "homography_failure",
    "semantic_ordering_failure",
]


class TiledPerspectiveLatticeFailure(ValueError):
    """Explicit S1-only failure surface, independent of the S0 contract."""

    def __init__(self, reason: S1FailureReason):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class LatticeTileCell:
    """A unique local quadrilateral with a topology-derived lattice address."""

    id: int
    cyclic_pixel_corners: Quad
    centroid_pixel: Point
    area_pixel: float
    lattice_coordinate: tuple[int, int] | None
    neighbor_ids_by_local_edge: tuple[int | None, int | None, int | None, int | None]


@dataclass(frozen=True)
class TileLatticeComponent:
    cell_ids: tuple[int, ...]
    coordinate_extents: tuple[int, int, int, int]

    @property
    def tile_count(self) -> int:
        return len(self.cell_ids)


@dataclass(frozen=True)
class RectangularLatticeCore:
    j0: int
    i0: int
    rows: int
    columns: int
    cell_ids: tuple[int, ...]


@dataclass(frozen=True)
class TiledPerspectiveLatticeResult:
    version: str
    decoded_width: int
    decoded_height: int
    cells: tuple[LatticeTileCell, ...]
    selected_component: TileLatticeComponent
    authoritative_core: RectangularLatticeCore
    h_lattice_to_tiled: Matrix3x3
    vp_a_homogeneous: Vector3
    vp_b_homogeneous: Vector3
    authoritative_quad_pixel: Quad
    authoritative_quad_source_normalized: Quad
    raw_quadrilateral_count: int
    deduplicated_cell_count: int
    coherent_component_count: int
    reprojection_mean_px: float
    reprojection_max_px: float


@dataclass(frozen=True)
class _CellState:
    coordinate: tuple[int, int]
    orientation_index: int


@dataclass(frozen=True)
class _NormalizedComponent:
    cell_ids: tuple[int, ...]
    states: dict[int, _CellState]
    core: RectangularLatticeCore


def read_tiled_perspective_lattice(encoded_image: bytes) -> TiledPerspectiveLatticeResult:
    """Read a coherent TileCell lattice directly from immutable TILED bytes."""
    if not encoded_image:
        raise TiledPerspectiveLatticeFailure("invalid_input_image")
    image = cv2.imdecode(np.frombuffer(encoded_image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise TiledPerspectiveLatticeFailure("invalid_input_image")
    return read_tiled_perspective_lattice_image(image)


def read_tiled_perspective_lattice_image(image: np.ndarray) -> TiledPerspectiveLatticeResult:
    """Enumerate local cells and fit a shared lattice homography."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise TiledPerspectiveLatticeFailure("invalid_input_image")
    height, width = image.shape[:2]
    if width < 2 or height < 2:
        raise TiledPerspectiveLatticeFailure("invalid_input_image")
    raw_candidates = _extract_s1_quad_candidates(image)
    return build_tiled_perspective_lattice(
        width=width,
        height=height,
        quadrilaterals=raw_candidates,
    )


def build_tiled_perspective_lattice(
    *,
    width: int,
    height: int,
    quadrilaterals: Sequence[np.ndarray | Sequence[Sequence[float]]],
) -> TiledPerspectiveLatticeResult:
    """Build S1 topology from complete local quadrilaterals.

    This direct kernel is intentionally public for deterministic synthetic
    tests.  Production reading simply supplies its local contour observations.
    """
    minimum_area = max(16.0, width * height * _MIN_AREA_FRACTION)
    raw_candidates = [
        _cyclic_quad(np.asarray(candidate, dtype=np.float64), minimum_area)
        for candidate in quadrilaterals
    ]
    if not raw_candidates:
        raise TiledPerspectiveLatticeFailure("no_complete_tile")
    unique_candidates = _deduplicate_quads(raw_candidates)
    cells, local_vertex_ids, neighbor_ids = _build_local_topology(unique_candidates)
    graph_components = _connected_cell_ids(neighbor_ids)

    normalized_components: list[_NormalizedComponent] = []
    for component_ids in graph_components:
        normalized = _normalize_component(component_ids, cells, local_vertex_ids, neighbor_ids)
        if normalized is not None:
            normalized_components.append(normalized)
    if not normalized_components:
        raise TiledPerspectiveLatticeFailure("lattice_assignment_conflict")

    winner = min(
        normalized_components,
        key=lambda component: _component_selection_key(component, cells),
    )
    homography, mean_residual, max_residual = _fit_shared_homography(
        winner,
        cells,
        width,
        height,
    )
    core = winner.core
    raw_core_quad = _map_lattice_points(
        homography,
        np.asarray(
            (
                (float(core.j0), float(core.i0)),
                (float(core.j0 + core.columns), float(core.i0)),
                (float(core.j0 + core.columns), float(core.i0 + core.rows)),
                (float(core.j0), float(core.i0 + core.rows)),
            ),
            dtype=np.float64,
        ),
    )
    if raw_core_quad is None:
        raise TiledPerspectiveLatticeFailure("homography_failure")
    authoritative = _semantic_core_quad(raw_core_quad, minimum_area)
    vp_a = homography @ np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    vp_b = homography @ np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    if not np.all(np.isfinite(vp_a)) or not np.all(np.isfinite(vp_b)):
        raise TiledPerspectiveLatticeFailure("homography_failure")

    states = {
        cell_id: state
        for component in normalized_components
        for cell_id, state in component.states.items()
    }
    public_cells = tuple(
        LatticeTileCell(
            id=cell_id,
            cyclic_pixel_corners=_as_quad(cells[cell_id]),
            centroid_pixel=(
                float(np.mean(cells[cell_id][:, 0])),
                float(np.mean(cells[cell_id][:, 1])),
            ),
            area_pixel=_polygon_area(cells[cell_id]),
            lattice_coordinate=states[cell_id].coordinate if cell_id in states else None,
            neighbor_ids_by_local_edge=tuple(neighbor_ids[cell_id]),  # type: ignore[arg-type]
        )
        for cell_id in range(len(cells))
    )
    extents = _coordinate_extents(winner.states)
    pixel_quad = _as_quad(authoritative)
    normalized_quad = _as_quad(
        np.column_stack((authoritative[:, 0] / width, authoritative[:, 1] / height))
    )
    return TiledPerspectiveLatticeResult(
        version="afc-sr1-tiled-perspective-reader/s1",
        decoded_width=width,
        decoded_height=height,
        cells=public_cells,
        selected_component=TileLatticeComponent(winner.cell_ids, extents),
        authoritative_core=core,
        h_lattice_to_tiled=_as_matrix3x3(homography),
        vp_a_homogeneous=_as_vector3(vp_a),
        vp_b_homogeneous=_as_vector3(vp_b),
        authoritative_quad_pixel=pixel_quad,
        authoritative_quad_source_normalized=normalized_quad,
        raw_quadrilateral_count=len(raw_candidates),
        deduplicated_cell_count=len(cells),
        coherent_component_count=len(normalized_components),
        reprojection_mean_px=mean_residual,
        reprojection_max_px=max_residual,
    )


def _extract_s1_quad_candidates(image: np.ndarray) -> list[np.ndarray]:
    """Observe complete local quads without S0 location or semantic gates."""
    height, width = image.shape[:2]
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        cv2.GaussianBlur(grayscale, (3, 3), 0),
        _S1_CANNY_LOW,
        _S1_CANNY_HIGH,
    )
    closed_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = max(16.0, width * height * _MIN_AREA_FRACTION)
    candidates: list[np.ndarray] = []
    for contour in contours:
        # A contour touching the decoded raster edge cannot establish all four
        # observed sides, unlike a merely border-adjacent complete cell.
        x, y, contour_width, contour_height = cv2.boundingRect(contour)
        if x == 0 or y == 0 or x + contour_width >= width or y + contour_height >= height:
            continue
        perimeter = cv2.arcLength(contour, True)
        if not np.isfinite(perimeter) or perimeter <= _GEOMETRY_EPSILON:
            continue
        approximate = cv2.approxPolyDP(
            contour,
            _CONTOUR_APPROXIMATION_RATIO * perimeter,
            True,
        ).reshape(-1, 2).astype(np.float64)
        if len(approximate) != 4:
            continue
        try:
            candidates.append(_cyclic_quad(approximate, minimum_area))
        except TiledPerspectiveReaderFailure:
            continue
    return candidates


def _cyclic_quad(points: np.ndarray, minimum_area: float) -> np.ndarray:
    """Validate and return a cyclic quad without imposing S0 near/far labels."""
    corners = np.asarray(points, dtype=np.float64)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise TiledPerspectiveReaderFailure("invalid_tile_geometry")
    for index in range(4):
        for other_index in range(index):
            if np.linalg.norm(corners[index] - corners[other_index]) <= _GEOMETRY_EPSILON:
                raise TiledPerspectiveReaderFailure("invalid_tile_geometry")
    centroid = np.mean(corners, axis=0)
    cyclic = corners[
        np.argsort(np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0]))
    ]
    if not _is_strictly_convex(cyclic, minimum_area):
        raise TiledPerspectiveReaderFailure("invalid_tile_geometry")
    return cyclic


def _deduplicate_quads(candidates: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Keep one deterministic representative of near-identical contour loops."""
    ordered = sorted(
        candidates,
        key=lambda corners: tuple(float(value) for value in corners.reshape(-1)),
    )
    unique: list[np.ndarray] = []
    for candidate in ordered:
        if not any(_same_quad(candidate, retained) for retained in unique):
            unique.append(candidate)
    return unique


def _same_quad(first: np.ndarray, second: np.ndarray) -> bool:
    for reverse in (False, True):
        comparison = second[::-1] if reverse else second
        for shift in range(4):
            rotated = np.roll(comparison, shift, axis=0)
            if np.all(
                np.linalg.norm(first - rotated, axis=1) <= _S1_DUPLICATE_VERTEX_TOLERANCE_PX
            ):
                return True
    return False


def _build_local_topology(
    cells: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]], list[list[int | None]]]:
    """Snap local vertices and create shared-boundary neighbor references."""
    vertices = [(cell_id, corner_id, point) for cell_id, cell in enumerate(cells) for corner_id, point in enumerate(cell)]
    parent = list(range(len(vertices)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(first: int, second: int) -> None:
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first in range(len(vertices)):
        for second in range(first):
            if np.linalg.norm(vertices[first][2] - vertices[second][2]) <= _S1_VERTEX_SNAP_TOLERANCE_PX:
                join(first, second)
    root_to_id: dict[int, int] = {}
    local_vertex_ids: list[list[int]] = [[-1] * 4 for _ in cells]
    for vertex_index, (cell_id, corner_id, _) in enumerate(vertices):
        component_root = root(vertex_index)
        root_to_id.setdefault(component_root, len(root_to_id))
        local_vertex_ids[cell_id][corner_id] = root_to_id[component_root]

    edge_groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell_id, vertex_ids in enumerate(local_vertex_ids):
        for edge_index, (start, end) in enumerate(_EDGE_CORNERS):
            key = tuple(sorted((vertex_ids[start], vertex_ids[end])))
            edge_groups.setdefault(key, []).append((cell_id, edge_index))
    neighbors: list[list[int | None]] = [[None, None, None, None] for _ in cells]
    for references in edge_groups.values():
        if len(references) == 2 and references[0][0] != references[1][0]:
            (first_cell, first_edge), (second_cell, second_edge) = references
            neighbors[first_cell][first_edge] = second_cell
            neighbors[second_cell][second_edge] = first_cell
    return list(cells), [tuple(ids) for ids in local_vertex_ids], neighbors


def _connected_cell_ids(neighbors: Sequence[Sequence[int | None]]) -> list[tuple[int, ...]]:
    pending = set(range(len(neighbors)))
    components: list[tuple[int, ...]] = []
    while pending:
        seed = min(pending)
        pending.remove(seed)
        visited = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            for neighbor in neighbors[current]:
                if neighbor is not None and neighbor not in visited:
                    visited.add(neighbor)
                    pending.discard(neighbor)
                    frontier.append(neighbor)
        components.append(tuple(sorted(visited)))
    return components


def _normalize_component(
    cell_ids: tuple[int, ...],
    cells: Sequence[np.ndarray],
    local_vertex_ids: Sequence[tuple[int, int, int, int]],
    neighbors: Sequence[Sequence[int | None]],
) -> _NormalizedComponent | None:
    """BFS one D4 orientation; discard the component on a path conflict."""
    seed = cell_ids[0]
    states: dict[int, _CellState] = {seed: _CellState((0, 0), 0)}
    frontier = [seed]
    while frontier:
        cell_id = frontier.pop(0)
        for edge_index, neighbor_id in enumerate(neighbors[cell_id]):
            if neighbor_id is None or neighbor_id not in cell_ids:
                continue
            neighbor_edge = next(
                index for index, item in enumerate(neighbors[neighbor_id]) if item == cell_id
            )
            possibilities = _neighbor_states(
                states[cell_id],
                edge_index,
                neighbor_edge,
                local_vertex_ids[cell_id],
                local_vertex_ids[neighbor_id],
            )
            if len(possibilities) != 1:
                return None
            expected = possibilities[0]
            existing = states.get(neighbor_id)
            if existing is None:
                if any(
                    other_id != neighbor_id and other_state.coordinate == expected.coordinate
                    for other_id, other_state in states.items()
                ):
                    return None
                states[neighbor_id] = expected
                frontier.append(neighbor_id)
            elif existing != expected:
                return None
    if len(states) != len(cell_ids):
        return None
    core = _largest_rectangular_core(states)
    return _NormalizedComponent(cell_ids, states, core)


def _neighbor_states(
    current: _CellState,
    current_edge: int,
    neighbor_edge: int,
    current_vertices: tuple[int, int, int, int],
    neighbor_vertices: tuple[int, int, int, int],
) -> list[_CellState]:
    """Enumerate the unique exterior D4 state implied by one shared edge."""
    current_offsets = _ORIENTATIONS[current.orientation_index]
    current_start, current_end = _EDGE_CORNERS[current_edge]
    target_coordinates = {
        current_vertices[current_start]: _add_coordinate(
            current.coordinate, current_offsets[current_start]
        ),
        current_vertices[current_end]: _add_coordinate(
            current.coordinate, current_offsets[current_end]
        ),
    }
    neighbor_start, neighbor_end = _EDGE_CORNERS[neighbor_edge]
    possibilities: list[_CellState] = []
    for orientation_index, offsets in enumerate(_ORIENTATIONS):
        first_vertex = neighbor_vertices[neighbor_start]
        second_vertex = neighbor_vertices[neighbor_end]
        if first_vertex not in target_coordinates or second_vertex not in target_coordinates:
            continue
        coordinate = _subtract_coordinate(target_coordinates[first_vertex], offsets[neighbor_start])
        if _add_coordinate(coordinate, offsets[neighbor_end]) != target_coordinates[second_vertex]:
            continue
        if coordinate != current.coordinate:
            possibilities.append(_CellState(coordinate, orientation_index))
    return possibilities


def _largest_rectangular_core(states: dict[int, _CellState]) -> RectangularLatticeCore:
    by_coordinate = {state.coordinate: cell_id for cell_id, state in states.items()}
    coordinates = sorted(by_coordinate)
    best: RectangularLatticeCore | None = None
    for j0, i0 in coordinates:
        for j1, i1 in coordinates:
            if j1 < j0 or i1 < i0:
                continue
            covered = [
                (j, i)
                for j in range(j0, j1 + 1)
                for i in range(i0, i1 + 1)
            ]
            if not all(coordinate in by_coordinate for coordinate in covered):
                continue
            candidate = RectangularLatticeCore(
                j0=j0,
                i0=i0,
                rows=i1 - i0 + 1,
                columns=j1 - j0 + 1,
                cell_ids=tuple(sorted(by_coordinate[coordinate] for coordinate in covered)),
            )
            if best is None or _core_key(candidate) < _core_key(best):
                best = candidate
    if best is None:
        raise TiledPerspectiveLatticeFailure("no_rectangular_core")
    return best


def _core_key(core: RectangularLatticeCore) -> tuple[int, int, int, int, int]:
    return (
        -(core.rows * core.columns),
        -min(core.rows, core.columns),
        core.i0,
        core.j0,
        len(core.cell_ids),
    )


def _component_selection_key(
    component: _NormalizedComponent,
    cells: Sequence[np.ndarray],
) -> tuple[int, int, int, float, float, tuple[int, ...]]:
    core = component.core
    centroid = np.mean(
        np.asarray([np.mean(cells[cell_id], axis=0) for cell_id in component.cell_ids]),
        axis=0,
    )
    return (
        -len(component.cell_ids),
        -(core.rows * core.columns),
        -min(core.rows, core.columns),
        float(centroid[1]),
        float(centroid[0]),
        component.cell_ids,
    )


def _fit_shared_homography(
    component: _NormalizedComponent,
    cells: Sequence[np.ndarray],
    width: int,
    height: int,
) -> tuple[np.ndarray, float, float]:
    observations: dict[tuple[int, int], list[np.ndarray]] = {}
    for cell_id in component.cell_ids:
        state = component.states[cell_id]
        offsets = _ORIENTATIONS[state.orientation_index]
        for corner, offset in zip(cells[cell_id], offsets, strict=True):
            observations.setdefault(_add_coordinate(state.coordinate, offset), []).append(corner)
    source = np.asarray(sorted(observations), dtype=np.float64)
    destination = np.asarray(
        [np.mean(observations[tuple(point)], axis=0) for point in source.astype(int)],
        dtype=np.float64,
    )
    if len(source) < 4:
        raise TiledPerspectiveLatticeFailure("homography_failure")
    homography, _ = cv2.findHomography(source, destination, method=0)
    if homography is None or homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
        raise TiledPerspectiveLatticeFailure("homography_failure")
    homography = np.asarray(homography, dtype=np.float64)
    scale = max(1.0, float(np.linalg.norm(homography, ord="fro")))
    if (
        np.linalg.matrix_rank(homography) != 3
        or abs(float(np.linalg.det(homography))) <= _HOMOGRAPHY_EPSILON * scale**3
    ):
        raise TiledPerspectiveLatticeFailure("homography_failure")

    residuals: list[float] = []
    for cell_id in component.cell_ids:
        state = component.states[cell_id]
        source_corners = np.asarray(
            [_add_coordinate(state.coordinate, offset) for offset in _ORIENTATIONS[state.orientation_index]],
            dtype=np.float64,
        )
        reprojected = _map_lattice_points(homography, source_corners)
        if reprojected is None:
            raise TiledPerspectiveLatticeFailure("homography_failure")
        residuals.extend(np.linalg.norm(reprojected - cells[cell_id], axis=1).tolist())
    if not residuals or not np.all(np.isfinite(residuals)):
        raise TiledPerspectiveLatticeFailure("homography_failure")
    return homography, float(np.mean(residuals)), float(np.max(residuals))


def _semantic_core_quad(raw_quad: np.ndarray, minimum_area: float) -> np.ndarray:
    """Map the strongest image-depth core-boundary pair to AFC semantics."""
    i_boundaries = (raw_quad[[0, 1]], raw_quad[[3, 2]])
    j_boundaries = (raw_quad[[1, 2]], raw_quad[[0, 3]])

    def depth_span(boundaries: tuple[np.ndarray, np.ndarray]) -> float:
        return abs(float(np.mean(boundaries[0][:, 1])) - float(np.mean(boundaries[1][:, 1])))

    # A D4-normalized lattice does not reserve either axis for image depth.
    # Ties retain the prior i-boundary convention deterministically.
    i_span = depth_span(i_boundaries)
    j_span = depth_span(j_boundaries)
    boundaries = j_boundaries if j_span > i_span + _GEOMETRY_EPSILON else i_boundaries
    first_boundary, second_boundary = boundaries
    first_mean_y = float(np.mean(first_boundary[:, 1]))
    second_mean_y = float(np.mean(second_boundary[:, 1]))
    if abs(first_mean_y - second_mean_y) <= _GEOMETRY_EPSILON:
        raise TiledPerspectiveLatticeFailure("semantic_ordering_failure")
    near_edge, far_edge = (
        (first_boundary, second_boundary)
        if first_mean_y > second_mean_y
        else (second_boundary, first_boundary)
    )
    near_left, near_right = sorted(near_edge, key=lambda point: (float(point[0]), float(point[1])))
    far_left, far_right = sorted(far_edge, key=lambda point: (float(point[0]), float(point[1])))
    semantic = np.asarray((near_left, near_right, far_right, far_left), dtype=np.float64)
    if not _is_strictly_convex(semantic, minimum_area):
        raise TiledPerspectiveLatticeFailure("semantic_ordering_failure")
    return semantic


def _coordinate_extents(states: dict[int, _CellState]) -> tuple[int, int, int, int]:
    coordinates = [state.coordinate for state in states.values()]
    return (
        min(coordinate[0] for coordinate in coordinates),
        max(coordinate[0] for coordinate in coordinates),
        min(coordinate[1] for coordinate in coordinates),
        max(coordinate[1] for coordinate in coordinates),
    )


def _add_coordinate(
    first: tuple[int, int],
    second: tuple[int, int],
) -> tuple[int, int]:
    return first[0] + second[0], first[1] + second[1]


def _subtract_coordinate(
    first: tuple[int, int],
    second: tuple[int, int],
) -> tuple[int, int]:
    return first[0] - second[0], first[1] - second[1]
