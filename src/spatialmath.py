"""
Spatial math utilities using Viam's rust utils library.
Provides orientation vector ↔ quaternion conversion and vector rotation.
Based on: https://github.com/viam-labs/apriltag/blob/main/src/spatialmath.py
"""

import ctypes
import math
import os
import sys
from ctypes import POINTER, Structure, c_double

import numpy as np


class _Quaternion(Structure):
    _fields_ = []


class _OrientationVector(Structure):
    _fields_ = []


class _Vector3(Structure):
    _fields_ = [("x", c_double), ("y", c_double), ("z", c_double)]


class _QuatArray(Structure):
    _fields_ = [("real", c_double), ("i", c_double), ("j", c_double), ("k", c_double)]


class _OVArray(Structure):
    _fields_ = [("o_x", c_double), ("o_y", c_double), ("o_z", c_double), ("theta", c_double)]


_lib = None


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib

    # Find the rust utils library.
    lib_path = None
    try:
        import viam.rpc
        rpc_dir = os.path.dirname(viam.rpc.__file__)
        for ext in (".so", ".dylib", ".dll"):
            candidate = os.path.join(rpc_dir, f"libviam_rust_utils{ext}")
            if os.path.exists(candidate):
                lib_path = candidate
                break
    except ImportError:
        pass

    if lib_path is None and hasattr(sys, "_MEIPASS"):
        for ext in (".so", ".dylib", ".dll"):
            candidate = os.path.join(sys._MEIPASS, "viam", "rpc", f"libviam_rust_utils{ext}")
            if os.path.exists(candidate):
                lib_path = candidate
                break

    if lib_path is None:
        raise RuntimeError("Could not find libviam_rust_utils")

    _lib = ctypes.CDLL(lib_path)

    _lib.new_quaternion.argtypes = (c_double, c_double, c_double, c_double)
    _lib.new_quaternion.restype = POINTER(_Quaternion)

    _lib.new_orientation_vector.argtypes = (c_double, c_double, c_double, c_double)
    _lib.new_orientation_vector.restype = POINTER(_OrientationVector)

    _lib.quaternion_from_orientation_vector.argtypes = (POINTER(_OrientationVector),)
    _lib.quaternion_from_orientation_vector.restype = POINTER(_Quaternion)

    _lib.quaternion_get_components.argtypes = (POINTER(_Quaternion),)
    _lib.quaternion_get_components.restype = POINTER(_QuatArray)

    _lib.quaternion_rotate_vector.argtypes = (POINTER(_Quaternion), POINTER(_Vector3))
    _lib.quaternion_rotate_vector.restype = POINTER(_Vector3)

    _lib.orientation_vector_from_quaternion.argtypes = (POINTER(_Quaternion),)
    _lib.orientation_vector_from_quaternion.restype = POINTER(_OrientationVector)

    _lib.orientation_vector_get_components.argtypes = (POINTER(_OrientationVector),)
    _lib.orientation_vector_get_components.restype = POINTER(_OVArray)

    _lib.free_quaternion_memory.argtypes = (POINTER(_Quaternion),)
    _lib.free_quaternion_memory.restype = None

    _lib.free_orientation_vector_memory.argtypes = (POINTER(_OrientationVector),)
    _lib.free_orientation_vector_memory.restype = None

    _lib.free_vector_memory.argtypes = (POINTER(_Vector3),)
    _lib.free_vector_memory.restype = None

    return _lib


def transform_points_with_pose(o_x: float, o_y: float, o_z: float, theta_deg: float,
                                tx: float, ty: float, tz: float,
                                points: np.ndarray) -> np.ndarray:
    """Transform 3D points using a Viam Pose (OrientationVector + translation).

    This uses Viam's rust utils to perform the exact same rotation as the Go RDK:
    OV → quaternion → rotate each point → add translation.

    Matches: rdk/spatialmath Compose(offset_dq, point_dq).Point()

    Args:
        o_x, o_y, o_z: OrientationVector components from Pose proto
        theta_deg: Theta from Pose proto (in degrees)
        tx, ty, tz: Translation from Pose proto (in mm)
        points: (N, 3) numpy array of points in source frame (mm)

    Returns:
        (N, 3) numpy array of points in destination frame (mm)
    """
    theta_rad = math.radians(theta_deg)

    lib = _get_lib()
    ov = lib.new_orientation_vector(c_double(o_x), c_double(o_y), c_double(o_z), c_double(theta_rad))
    q = lib.quaternion_from_orientation_vector(ov)
    lib.free_orientation_vector_memory(ov)

    result = np.empty_like(points)
    for i in range(len(points)):
        v = _Vector3(float(points[i, 0]), float(points[i, 1]), float(points[i, 2]))
        rv = lib.quaternion_rotate_vector(q, ctypes.byref(v))
        result[i] = [rv.contents.x + tx, rv.contents.y + ty, rv.contents.z + tz]
        lib.free_vector_memory(rv)

    lib.free_quaternion_memory(q)
    return result
