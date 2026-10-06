"""Import the exported carveout with the CUDA driver API (libcuda, ctypes), map it, write and read it back. No toolkit needed."""
import ctypes, os, sys, time
cu = ctypes.CDLL("libcuda.so.1")
def ck(r, what):
    if r: s = ctypes.c_char_p(); cu.cuGetErrorString(r, ctypes.byref(s)); raise RuntimeError(f"{what}: {s.value.decode()} ({r})")
fd, size = int(os.environ["TF_CARVEOUT_FD"]), int(os.environ["TF_CARVEOUT_SIZE"])
ck(cu.cuInit(0), "cuInit"); dev = ctypes.c_int(); ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "device")
ctx = ctypes.c_void_p(); ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "ctx"); ck(cu.cuCtxSetCurrent(ctx), "set ctx")
h = ctypes.c_ulonglong(); ck(cu.cuMemImportFromShareableHandle(ctypes.byref(h), ctypes.c_void_p(fd), 1), "import")   # 1 = POSIX fd
va = ctypes.c_ulonglong(); ck(cu.cuMemAddressReserve(ctypes.byref(va), ctypes.c_size_t(size), ctypes.c_size_t(2 << 20), ctypes.c_ulonglong(0), ctypes.c_ulonglong(0)), "reserve")
ck(cu.cuMemMap(va, ctypes.c_size_t(size), ctypes.c_size_t(0), h, ctypes.c_ulonglong(0)), "map"); ck(cu.cuMemRelease(h), "release")
class Access(ctypes.Structure): _fields_ = [("loc_type", ctypes.c_int), ("loc_id", ctypes.c_int), ("flags", ctypes.c_ulonglong)]
acc = Access(1, 0, 3)    # device 0, read/write
ck(cu.cuMemSetAccess(va, ctypes.c_size_t(size), ctypes.byref(acc), ctypes.c_size_t(1)), "access")
t = time.time(); ck(cu.cuMemsetD8_v2(va, ctypes.c_ubyte(0x5A), ctypes.c_size_t(size)), "memset"); ck(cu.cuCtxSynchronize(), "sync"); dt = time.time() - t
buf = (ctypes.c_ubyte * 4096)()
for off in (0, size // 2, size - 4096):
    ck(cu.cuMemcpyDtoH_v2(buf, ctypes.c_ulonglong(va.value + off), ctypes.c_size_t(4096)), "copy"); assert all(b == 0x5A for b in buf), off
print(f"carveout mapped at 0x{va.value:x}: {size >> 20} MiB written ({size / dt / 1e9:.0f} GB/s memset) and read back OK")
