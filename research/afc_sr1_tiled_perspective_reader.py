"""AFC-SR1 S0 reader for one complete generated TILED cell.

This module deliberately treats the decoded TILED raster as its only
perspective authority.  It finds local contrast-defined quadrilateral cells;
it does not infer room extent, select global line families, or transfer
evidence from another image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

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
