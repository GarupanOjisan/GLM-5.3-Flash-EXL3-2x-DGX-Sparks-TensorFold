#!/usr/bin/env python3
"""Export the GB10 display carveout (DISPLAY_FRM, 2 GiB the firmware reserves and nothing on GB10 uses) as an NVIDIA RM
object on an inheritable /dev/nvidiactl fd that CUDA imports with cuMemImportFromShareableHandle (POSIX fd).
Needs CAP_SYS_ADMIN. A port of Enntity/atlas crates/spark-server/src/display_carveout/rm.rs (AGPL-3.0-only; layouts
from open-gpu-kernel-modules, MIT; checked for driver 580.173.02 / 580.178.04).

  carveout_export.py                -> prints base, size, and keeps the fd open while it sleeps (for a test)
  carveout_export.py -- CMD ARGS..  -> execs CMD with TF_CARVEOUT_FD / TF_CARVEOUT_SIZE set and the fd inherited"""
import ctypes, fcntl, os, sys

IOCTL_RM_ALLOC, IOCTL_RM_CONTROL, IOCTL_CHECK_VERSION_STR, IOCTL_REGISTER_FD = 0xC020462B, 0xC020462A, 0xC04846D2, 0xC00446C9
NV01_ROOT_CLIENT, NV01_DEVICE_0, NV20_SUBDEVICE_0, NV01_MEMORY_LIST_SYSTEM = 0x41, 0x80, 0x2080, 0x81
CMD_FB_GET_CARVEOUT_REGION_INFO, CMD_OS_UNIX_EXPORT_OBJECT_TO_FD = 0x20801360, 0x3D05
EXPORT_OBJECT_TYPE_RM, CARVEOUT_DISPLAY_FRM, OS02_CONTIGUOUS_CACHED, VERSION_CMD_QUERY = 1, 0, 0x1000, ord("2")
H_DEVICE, H_SUBDEVICE, H_LIST = 0x5C000001, 0x5C000002, 0x5C000010
VALIDATED = ("580.173.02", "580.178.04")
u32, u64, i32 = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int32

class ApiVersion(ctypes.Structure): _fields_ = [("cmd", u32), ("reply", u32), ("version", ctypes.c_char * 64)]
class Alloc(ctypes.Structure): _fields_ = [("h_root", u32), ("h_parent", u32), ("h_new", u32), ("h_class", u32), ("params", u64), ("params_size", u32), ("status", u32)]
class Control(ctypes.Structure): _fields_ = [("h_client", u32), ("h_object", u32), ("cmd", u32), ("flags", u32), ("params", u64), ("params_size", u32), ("status", u32)]
class Region(ctypes.Structure): _fields_ = [("base", u64), ("size", u64), ("kind", u32), ("_pad", u32)]
class RegionInfo(ctypes.Structure): _fields_ = [("count", u32), ("_pad", u32), ("regions", Region * 8)]
class MemoryList(ctypes.Structure):
    _fields_ = [(n, u32) for n in ("h_client", "h_parent", "h_object", "h_hw_res_client", "h_hw_res_device", "h_hw_res_handle", "pte_adjust", "reserved_0",
                                   "kind", "flags", "attr", "attr2", "height", "width", "format", "comprcovg", "zcullcovg", "page_count", "heap_owner", "_pad0")] + \
               [("guest_id", u64), ("range_begin", u64), ("range_end", u64), ("pitch", u32), ("ctag_offset", u32), ("size", u64), ("align", u64),
                ("page_number_list", u64), ("limit", u64), ("flags_os02", u32), ("_pad1", u32)]
class ExportToFd(ctypes.Structure): _fields_ = [("kind", u32), ("h_device", u32), ("h_parent", u32), ("h_object", u32), ("fd", i32), ("flags", u32)]
assert ctypes.sizeof(ApiVersion) == 72 and ctypes.sizeof(Alloc) == 32 and ctypes.sizeof(Control) == 32 and ctypes.sizeof(RegionInfo) == 200
assert ctypes.sizeof(MemoryList) == 152 and MemoryList.page_count.offset == 68 and MemoryList.size.offset == 112 and MemoryList.page_number_list.offset == 128 \
    and MemoryList.limit.offset == 136 and MemoryList.flags_os02.offset == 144 and ctypes.sizeof(ExportToFd) == 24 and ExportToFd.fd.offset == 16

def ioctl(fd, req, obj, what):
    try: fcntl.ioctl(fd, req, obj)
    except OSError as e: raise RuntimeError(f"{what}: {e}")

class Client:
    def __init__(self):
        self.ctl = os.open("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC); gpu = os.open("/dev/nvidia0", os.O_RDWR | os.O_CLOEXEC)
        v = ApiVersion(cmd=VERSION_CMD_QUERY); ioctl(self.ctl, IOCTL_CHECK_VERSION_STR, v, "RM version")
        reg = i32(self.ctl); ioctl(gpu, IOCTL_REGISTER_FD, reg, "register fd")
        self.handle = 0; self.handle = self.alloc(0, 0, NV01_ROOT_CLIENT, None, "client")
        self.alloc(self.handle, H_DEVICE, NV01_DEVICE_0, (ctypes.c_uint8 * 56)(), "device")      # NV0080_ALLOC_PARAMETERS, deviceId 0
        self.alloc(H_DEVICE, H_SUBDEVICE, NV20_SUBDEVICE_0, u32(0), "subdevice")                   # NV2080_ALLOC_PARAMETERS, subDeviceId 0
    def alloc(self, parent, handle, cls, params, what):
        a = Alloc(h_root=self.handle, h_parent=parent, h_new=handle, h_class=cls,
                  params=ctypes.addressof(params) if params is not None else 0, params_size=ctypes.sizeof(params) if params is not None else 0)
        ioctl(self.ctl, IOCTL_RM_ALLOC, a, what)
        if a.status: raise RuntimeError(f"RM alloc {what}: status 0x{a.status:x}")
        return a.h_new
    def control(self, obj, cmd, params, what):
        c = Control(h_client=self.handle, h_object=obj, cmd=cmd, params=ctypes.addressof(params), params_size=ctypes.sizeof(params))
        ioctl(self.ctl, IOCTL_RM_CONTROL, c, what)
        if c.status: raise RuntimeError(f"RM control {what}: status 0x{c.status:x}")

def export():
    line = open("/proc/driver/nvidia/version").readline()
    if not any(v in line.split() for v in VALIDATED): raise RuntimeError(f"driver not validated: {line.strip()}")
    cl = Client(); info = RegionInfo(); cl.control(H_SUBDEVICE, CMD_FB_GET_CARVEOUT_REGION_INFO, info, "carveout info")
    frm = next((r for r in info.regions[:min(info.count, 8)] if r.kind == CARVEOUT_DISPLAY_FRM and r.size > 0), None)
    if frm is None: raise RuntimeError("the driver reports no DISPLAY_FRM carveout")
    base, size = frm.base, frm.size
    pfn = u64(base >> 12)
    lst = MemoryList(page_count=1, size=size, limit=size - 1, page_number_list=ctypes.addressof(pfn), flags_os02=OS02_CONTIGUOUS_CACHED)
    cl.alloc(H_DEVICE, H_LIST, NV01_MEMORY_LIST_SYSTEM, lst, "memory list")
    fd = os.open("/dev/nvidiactl", os.O_RDWR); os.set_inheritable(fd, True)   # survives the exec
    ex = ExportToFd(kind=EXPORT_OBJECT_TYPE_RM, h_device=H_DEVICE, h_parent=H_DEVICE, h_object=H_LIST, fd=fd)
    try: cl.control(cl.handle, CMD_OS_UNIX_EXPORT_OBJECT_TO_FD, ex, "export")
    except Exception: os.close(fd); raise
    return fd, base, size

def drop_sys_admin():
    """CAP_SYS_ADMIN out of the bounding, ambient, effective, permitted and inheritable sets (the command keeps the rest)."""
    libc = ctypes.CDLL(None, use_errno=True); CAP = 21
    libc.prctl(47, 4, 0, 0, 0)                                   # PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL
    if libc.prctl(24, CAP, 0, 0, 0): raise RuntimeError(f"PR_CAPBSET_DROP: {os.strerror(ctypes.get_errno())}")
    class Hdr(ctypes.Structure): _fields_ = [("version", u32), ("pid", i32)]
    class Data(ctypes.Structure): _fields_ = [("effective", u32), ("permitted", u32), ("inheritable", u32)]
    hdr = Hdr(0x20080522, 0); data = (Data * 2)()
    if libc.capget(ctypes.byref(hdr), data): raise RuntimeError(f"capget: {os.strerror(ctypes.get_errno())}")
    for f in ("effective", "permitted", "inheritable"): setattr(data[0], f, getattr(data[0], f) & ~(1 << CAP))
    if libc.capset(ctypes.byref(hdr), data): raise RuntimeError(f"capset: {os.strerror(ctypes.get_errno())}")

if __name__ == "__main__":
    os.environ.pop("TF_CARVEOUT_FD", None); os.environ.pop("TF_CARVEOUT_SIZE", None)
    try:
        fd, base, size = export()
        print(f"[carveout] display carveout: base 0x{base:x}, {size >> 20} MiB, exported as fd {fd}", flush=True)
        os.environ["TF_CARVEOUT_FD"] = str(fd); os.environ["TF_CARVEOUT_SIZE"] = str(size)
    except Exception as e:
        print(f"[carveout] not lent: {e}; running without it", flush=True)
    if len(sys.argv) > 2 and sys.argv[1] == "--":
        drop_sys_admin(); os.execvp(sys.argv[2], sys.argv[2:])
    import time; time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 3)
