"""The GB10 display carveout as cache memory (experiment; after Enntity/sparkglm's DISPLAY_CARVEOUT=1, AGPL-3.0-only).

/opt/carveout_export.py (TF_GLM_CARVEOUT=1, CAP_SYS_ADMIN at start) exports the firmware's 2 GiB display reservation as
an RM object on an inherited fd (TF_CARVEOUT_FD, TF_CARVEOUT_SIZE). Here it is imported once (cuMemImportFromShareableHandle,
cuMemMap) and handed out as whole uint8 tensors, first come first served; nothing is ever given back. The GPU maps it
L2-uncached (data read twice costs DRAM twice), so only the latent KV planes go there (``forward.Caches``). Replies do not
change: placement moves allocations only."""
from __future__ import annotations

import ctypes
import os

import torch

_mapped: tuple[int, int] | None = None     # (device pointer, bytes): the whole carveout, mapped once
_cu = None
_used = 0
_ALIGN = 2 << 20
_EXTRA = 4 << 30                           # address space reserved past the carveout for the plane that straddles its end


def size() -> int:
    """Bytes the carveout offers (0: none exported to this process)."""
    try:
        return int(os.environ.get("TF_CARVEOUT_SIZE", "0")) if os.environ.get("TF_CARVEOUT_FD") else 0
    except ValueError:
        return 0


def _map() -> tuple[int, int] | None:
    global _mapped, _cu
    if _mapped is not None:
        return _mapped
    fd, nbytes = os.environ.get("TF_CARVEOUT_FD"), size()
    if not fd or not nbytes:
        return None
    torch.cuda.init()
    torch.empty(1, device="cuda")              # the primary context exists
    cu = ctypes.CDLL("libcuda.so.1")

    def ck(r, what):
        if r:
            s = ctypes.c_char_p()
            cu.cuGetErrorString(r, ctypes.byref(s))
            raise RuntimeError(f"carveout {what}: {s.value.decode() if s.value else r}")
    dev = ctypes.c_int()
    ck(cu.cuDeviceGet(ctypes.byref(dev), torch.cuda.current_device()), "device")
    ctx = ctypes.c_void_p()
    ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "context")
    ck(cu.cuCtxSetCurrent(ctx), "set context")
    h = ctypes.c_ulonglong()
    ck(cu.cuMemImportFromShareableHandle(ctypes.byref(h), ctypes.c_void_p(int(fd)), 1), "import")   # 1: POSIX fd
    os.close(int(fd))
    va = ctypes.c_ulonglong()
    ck(cu.cuMemAddressReserve(ctypes.byref(va), ctypes.c_size_t(nbytes + _EXTRA), ctypes.c_size_t(_ALIGN),
                              ctypes.c_ulonglong(0), ctypes.c_ulonglong(0)), "reserve")
    ck(cu.cuMemMap(va, ctypes.c_size_t(nbytes), ctypes.c_size_t(0), h, ctypes.c_ulonglong(0)), "map")
    ck(cu.cuMemRelease(h), "release handle")

    class Access(ctypes.Structure):
        _fields_ = [("loc_type", ctypes.c_int), ("loc_id", ctypes.c_int), ("flags", ctypes.c_ulonglong)]
    acc = Access(1, dev.value, 3)              # this device, read and write
    ck(cu.cuMemSetAccess(va, ctypes.c_size_t(nbytes), ctypes.byref(acc), ctypes.c_size_t(1)), "access")
    _mapped, _cu = (va.value, nbytes), cu
    os.environ.pop("TF_CARVEOUT_FD", None)
    return _mapped


def _ck(r, what):
    if r:
        s = ctypes.c_char_p()
        _cu.cuGetErrorString(r, ctypes.byref(s))
        raise RuntimeError(f"carveout {what}: {s.value.decode() if s.value else r}")


def _extend(end: int) -> None:
    """Device memory (cuMemCreate) mapped right after the carveout, up to address offset ``end``, so one plane can run
    from the carveout's last bytes into ordinary memory as a single address range."""
    cu = _cu
    base, total = _mapped
    dev = ctypes.c_int()
    _ck(cu.cuCtxGetDevice(ctypes.byref(dev)), "device")

    class Prop(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("handle_types", ctypes.c_int), ("loc_type", ctypes.c_int),
                    ("loc_id", ctypes.c_int), ("win32", ctypes.c_void_p), ("flags", ctypes.c_uint64)]
    prop = Prop(1, 0, 1, dev.value, None, 0)          # pinned device memory on this device
    gran = ctypes.c_size_t()
    _ck(cu.cuMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), 0), "granularity")
    g = max(gran.value, _ALIGN)
    tail = -(-(end - total) // g) * g
    if tail > _EXTRA:
        raise RuntimeError(f"carveout: a {tail >> 20} MiB tail past the carveout, more than the {_EXTRA >> 20} reserved")
    h = ctypes.c_ulonglong()
    _ck(cu.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(tail), ctypes.byref(prop), ctypes.c_ulonglong(0)), "create (tail)")
    _ck(cu.cuMemMap(ctypes.c_ulonglong(base + total), ctypes.c_size_t(tail), ctypes.c_size_t(0), h, ctypes.c_ulonglong(0)),
        "map (tail)")
    _ck(cu.cuMemRelease(h), "release (tail)")

    class Access(ctypes.Structure):
        _fields_ = [("loc_type", ctypes.c_int), ("loc_id", ctypes.c_int), ("flags", ctypes.c_ulonglong)]
    acc = Access(1, dev.value, 3)
    _ck(cu.cuMemSetAccess(ctypes.c_ulonglong(base + total), ctypes.c_size_t(tail), ctypes.byref(acc), ctypes.c_size_t(1)),
        "access (tail)")


class _View:
    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3,
                                         "strides": None}


def take(shape, dtype=torch.uint8) -> torch.Tensor | None:
    """A zeroed tensor of ``shape``: in the carveout while it fits, the one that runs past its end gets the carveout's
    remainder followed by device memory in one address range; None once the carveout is used up."""
    global _used
    m = _map()
    if m is None:
        return None
    n = 1
    for d in shape:
        n *= int(d)
    nbytes = n * torch.empty(0, dtype=dtype).element_size()
    start = -(-_used // _ALIGN) * _ALIGN
    left = m[1] - start
    if left <= 0:
        return None
    if nbytes > left:                          # the last plane: the carveout's remainder, then device memory after it
        _extend(start + nbytes)
    ptr, _used = m[0] + start, min(start + nbytes, m[1])
    if nbytes > left:
        _used = m[1]
    t = torch.as_tensor(_View(ptr, nbytes), device="cuda").view(dtype).view(*shape)
    t.zero_()
    return t


def used() -> int:
    return _used
