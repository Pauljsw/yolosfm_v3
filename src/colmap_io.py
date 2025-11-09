"""
COLMAP Binary Format Parser

Reads COLMAP binary reconstruction files:
- cameras.bin
- images.bin
- points3D.bin

Based on COLMAP file format specification.
"""
import numpy as np
import struct
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple
import logging

logger = logging.getLogger(__name__)


class Camera(NamedTuple):
    """Camera model"""
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # fx, fy, cx, cy, ...


class Image(NamedTuple):
    """Image with pose and observations"""
    id: int
    qvec: np.ndarray  # Quaternion (w, x, y, z)
    tvec: np.ndarray  # Translation (x, y, z)
    camera_id: int
    name: str
    xys: np.ndarray  # 2D keypoint coordinates (N, 2)
    point3D_ids: np.ndarray  # Corresponding 3D point IDs (N,)


class Point3D(NamedTuple):
    """3D point with track"""
    id: int
    xyz: np.ndarray  # 3D coordinates
    rgb: np.ndarray  # RGB color (uint8)
    error: float
    image_ids: List[int]  # Images observing this point
    point2D_idxs: List[int]  # Index in each image's keypoint list


# Camera model mappings
CAMERA_MODELS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE"
}

CAMERA_MODEL_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12
}


def read_cameras_binary(path: str) -> Dict[int, Camera]:
    """
    Read cameras.bin file.

    Args:
        path: Path to cameras.bin

    Returns:
        Dictionary mapping camera_id to Camera
    """
    cameras = {}

    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]

            model_name = CAMERA_MODELS.get(model_id, f"UNKNOWN_{model_id}")
            num_params = CAMERA_MODEL_NUM_PARAMS.get(model_name, 0)

            params = np.array(struct.unpack(f"<{num_params}d", f.read(8 * num_params)))

            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=params
            )

    logger.info(f"Read {len(cameras)} cameras from {path}")
    return cameras


def read_images_binary(path: str) -> Dict[int, Image]:
    """
    Read images.bin file.

    Args:
        path: Path to images.bin

    Returns:
        Dictionary mapping image_id to Image
    """
    images = {}

    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)))
            tvec = np.array(struct.unpack("<3d", f.read(24)))
            camera_id = struct.unpack("<I", f.read(4))[0]

            # Read image name (null-terminated string)
            name_chars = []
            while True:
                char = f.read(1)
                if char == b'\x00':
                    break
                name_chars.append(char)
            name = b''.join(name_chars).decode('utf-8')

            # Read 2D points
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            xys = np.zeros((num_points2D, 2), dtype=np.float64)
            point3D_ids = np.zeros(num_points2D, dtype=np.int64)

            for i in range(num_points2D):
                x = struct.unpack("<d", f.read(8))[0]
                y = struct.unpack("<d", f.read(8))[0]
                point3D_id = struct.unpack("<Q", f.read(8))[0]

                xys[i] = [x, y]
                point3D_ids[i] = point3D_id if point3D_id != np.iinfo(np.uint64).max else -1

            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                xys=xys,
                point3D_ids=point3D_ids
            )

    logger.info(f"Read {len(images)} images from {path}")
    return images


def read_points3D_binary(path: str) -> Dict[int, Point3D]:
    """
    Read points3D.bin file.

    Args:
        path: Path to points3D.bin

    Returns:
        Dictionary mapping point3D_id to Point3D
    """
    points3D = {}

    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_points):
            point3D_id = struct.unpack("<Q", f.read(8))[0]
            xyz = np.array(struct.unpack("<3d", f.read(24)))
            rgb = np.array(struct.unpack("<3B", f.read(3)), dtype=np.uint8)
            error = struct.unpack("<d", f.read(8))[0]

            # Read track
            track_length = struct.unpack("<Q", f.read(8))[0]
            image_ids = []
            point2D_idxs = []

            for _ in range(track_length):
                image_id = struct.unpack("<I", f.read(4))[0]
                point2D_idx = struct.unpack("<I", f.read(4))[0]
                image_ids.append(image_id)
                point2D_idxs.append(point2D_idx)

            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs
            )

    logger.info(f"Read {len(points3D)} 3D points from {path}")
    return points3D


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.

    Args:
        qvec: Quaternion (w, x, y, z)

    Returns:
        3x3 rotation matrix
    """
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec

    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

    return R


def image_to_world(qvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert image pose to world-to-camera transformation.

    COLMAP stores: image = R * world + t (world-to-camera)

    Args:
        qvec: Quaternion (w, x, y, z)
        tvec: Translation vector (x, y, z)

    Returns:
        (R, t): Rotation matrix and translation vector
    """
    R = qvec2rotmat(qvec)
    t = tvec.reshape(3, 1)

    return R, t


if __name__ == '__main__':
    # Test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.colmap_io <sparse_dir>")
        sys.exit(1)

    sparse_dir = Path(sys.argv[1])

    cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
    images = read_images_binary(str(sparse_dir / "images.bin"))
    points3D = read_points3D_binary(str(sparse_dir / "points3D.bin"))

    print(f"\n✅ Successfully parsed COLMAP reconstruction:")
    print(f"   Cameras: {len(cameras)}")
    print(f"   Images: {len(images)}")
    print(f"   3D Points: {len(points3D)}")

    # Show first camera
    if cameras:
        cam = list(cameras.values())[0]
        print(f"\n   First camera:")
        print(f"     Model: {cam.model}")
        print(f"     Size: {cam.width}×{cam.height}")
        print(f"     Params: {cam.params}")

    # Show first image
    if images:
        img = list(images.values())[0]
        print(f"\n   First image:")
        print(f"     Name: {img.name}")
        print(f"     Camera ID: {img.camera_id}")
        print(f"     2D Points: {len(img.xys)}")
        print(f"     Quat: {img.qvec}")
        print(f"     Trans: {img.tvec}")
