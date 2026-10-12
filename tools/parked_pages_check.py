#!/usr/bin/env python3
"""Checks for patch 0119 (the startup budget counts the pages parked on the kernel's per-CPU free lists), run inside
the image that scripts/prepare.sh built (CPU only: no GPU, no model, about a second):

    docker run --rm --entrypoint python -v "$PWD/tools/parked_pages_check.py:/c.py" tensorfold-glm53:v0.6.0 /c.py

or on any machine against TensorFold's patched source (no torch needed):

    PYTHONPATH=/path/to/TensorFold/src python tools/parked_pages_check.py

Synthetic /proc/zoneinfo and /proc/meminfo files (capacity.ZONEINFO / MEMINFO point at them) and a stand-in for torch:
page lists on several CPUs, zones and nodes (only the ``count:`` lines under ``pagesets`` are summed), zones without
lists, no lists at all, an unreadable or garbled file (0, never an error), and the figures the budget reads from them
(``_meminfo``, ``available_bytes``, ``page_room``, on a unified and on a discrete GPU) with the parked pages in them.
Exit code 1 when a check fails, and on an image without the patch.
"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

from tensorfold.cuda import capacity

GIB, KIB = 1024**3, 1024
PAGE = os.sysconf("SC_PAGE_SIZE")
failures = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f" ({detail})" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def cpus(counts, indent=4):
    """The page lists of one zone: a ``cpu:`` block a count, as the kernel prints them."""
    pad = " " * indent
    return "".join(f"{pad}cpu: {n}\n{pad}          count:    {c}\n{pad}          high:     5000\n{pad}          batch:    63\n"
                   f"{pad}          high_min: 2146\n{pad}          high_max: 198283\n  vm stats threshold: 110\n"
                   for n, c in enumerate(counts))


def zone(node, name, counts=None, managed=1000):
    """One zone's block: stats lines (one with a ``count`` in its name), the page lists when ``counts`` is given, start_pfn."""
    text = f"Node {node}, zone {name:>8}\n  pages free     10\n        managed  {managed}\n      nr_free_cma  0\n"
    if counts is not None:
        text += "  pagesets\n" + cpus(counts)
    return text + "  start_pfn:           4096\n"


def run(zoneinfo, meminfo, torch=None, zone_path=None):
    """The figures the budget reads with those files in place of /proc's."""
    d = Path(tempdir.name)
    if zoneinfo is not None:
        (d / "zoneinfo").write_text(zoneinfo)
    (d / "meminfo").write_text(meminfo)
    capacity.ZONEINFO = zone_path or str(d / "zoneinfo")
    capacity.MEMINFO = str(d / "meminfo")
    return NS(parked=parked(), info=capacity._meminfo(),
              avail=capacity.available_bytes(torch) if torch else None, room=capacity.page_room(torch) if torch else None)


def fake(integrated, free=0, total=0):
    props = NS(is_integrated=integrated)
    return NS(cuda=NS(mem_get_info=lambda: (free, total), get_device_properties=lambda i: props))


def meminfo(total_gib, avail_gib):
    return (f"MemTotal:       {total_gib * GIB // KIB} kB\nMemFree:        {GIB // KIB} kB\n"
            f"MemAvailable:   {int(avail_gib * GIB) // KIB} kB\nBuffers:        1 kB\n")


def parked():
    return capacity.parked_bytes() if hasattr(capacity, "parked_bytes") else None      # an image without the patch


tempdir = tempfile.TemporaryDirectory()
os.environ.pop("TENSORFOLD_MEMORY_RESERVE_GIB", None)
check("capacity.parked_bytes exists (patch 0119 applied)", hasattr(capacity, "parked_bytes"))

MEM = meminfo(128, 100)
unified = fake(True, free=10 * GIB, total=128 * GIB)
discrete = fake(False, free=60 * GIB, total=80 * GIB)
reserve = capacity.reserve_bytes(128 * GIB, host=True)             # a tenth of the host's memory (12.8 GiB)

# several CPUs on several zones of two nodes: every count, and nothing else
counts_a, counts_b, counts_c = [0, 3143, 2332, 17], [5, 5, 5], [100000]
text = (zone(0, "DMA", counts_a) + zone(0, "DMA32", None) + zone(0, "Normal", counts_b) + zone(1, "Normal", counts_c)
        + zone(1, "Movable", None))
total = sum(counts_a + counts_b + counts_c)
r = run(text, MEM, unified)
check("sums the counts of every CPU, zone and node", r.parked == total * PAGE, f"{r.parked} != {total * PAGE}")
check("MemAvailable includes them", r.info["MemAvailable"] == 100 * GIB + total * PAGE)
check("MemTotal is the file's", r.info["MemTotal"] == 128 * GIB)
check("available_bytes (unified GPU) is MemAvailable + parked - the 10 % reserve",
      r.avail == 100 * GIB + total * PAGE - reserve, f"{r.avail}")
check("page_room (unified GPU) is MemAvailable + parked", r.room == 100 * GIB + total * PAGE)
r = run(text, MEM, discrete)
check("a discrete GPU is bounded by the smaller of both: here its own free memory",
      r.avail == 60 * GIB - capacity.reserve_bytes(80 * GIB), f"{r.avail}")
check("page_room is None on a discrete GPU", r.room is None)

# gibibytes of them, like after an image copy: 6 GiB parked is 6 GiB more budget
six = 6 * GIB // PAGE
r = run(zone(0, "Normal", [six // 4] * 4), meminfo(128, 90), unified)
check("6 GiB parked raise the budget by 6 GiB",
      r.avail - (90 * GIB - reserve) == six // 4 * 4 * PAGE == 6 * GIB, f"{r.avail}")

# other lines named count, and the lines after the lists, are not page lists
noise = (zone(0, "Normal", [10]).replace("  start_pfn:           4096\n", "  start_pfn:           4096\n      count:    99999\n")
         + "Node 0, zone Other\n      count:    77777\n  pagesets\n    cpu: 0\n              high:     1\n")
r = run("  count: 5\n      count: 6\n" + noise, MEM)
check("only the count: lines under pagesets", r.parked == 10 * PAGE, f"{r.parked}")

# the kernel's other layouts of the lists
r = run("Node 0, zone Normal\n  pagesets\n    cpu: 0\n              count: 7\n              high: 9\n"
        "  vm stats threshold: 8\n    cpu: 1\n              count: 8\n  all_unreclaimable: 0\n", MEM)
check("a list that ends at the next field", r.parked == 15 * PAGE, f"{r.parked}")

# no page lists, an empty, an unreadable and a garbled file: nothing parked, the start goes on
no_lists = zone(0, "DMA", None) + zone(0, "Normal", None)
for label, text_, path in (("no page lists", no_lists, None), ("an empty file", "", None),
                           ("a missing file", None, str(Path(tempdir.name) / "none")),
                           ("a directory", None, tempdir.name),
                           ("a count that is not a number", zone(0, "Normal", [1]).replace("count:    1", "count:    x"), None),
                           ("a count with no value", zone(0, "Normal", [1]).replace("count:    1", "count:"), None),
                           ("binary", "\x00\xff\xfe" * 10, None)):
    r = run(text_, MEM, unified, zone_path=path)
    check(f"{label}: nothing parked, MemAvailable as read",
          r.parked == 0 and r.info["MemAvailable"] == 100 * GIB and r.avail == 100 * GIB - reserve, f"{r.parked}")
(Path(tempdir.name) / "zoneinfo").write_bytes(b"Node 0, zone Normal\n  pagesets\n    cpu: 0\n  count: \xff\xfe\n")
capacity.ZONEINFO = str(Path(tempdir.name) / "zoneinfo")
check("undecodable bytes: nothing parked", parked() == 0)

# the sum cannot pass the host's memory
r = run(zone(0, "Normal", [200 * GIB // PAGE]), meminfo(128, 120), unified)
check("MemAvailable is capped at MemTotal", r.info["MemAvailable"] == 128 * GIB)
# a /proc/meminfo that lacks a field is still unusable, and unreadable ones too
check("meminfo without MemAvailable: None", run(text, "MemTotal: 1 kB\n").info is None)
check("meminfo that is not numbers: None", run(text, "MemTotal: x kB\nMemAvailable: 1 kB\n").info is None)

# this host's real files, when there are any: the sum against awk's reading of them
capacity.ZONEINFO, capacity.MEMINFO = "/proc/zoneinfo", "/proc/meminfo"
try:
    real = sum(int(row.split()[1]) for row in Path("/proc/zoneinfo").read_text().splitlines()
               if row.startswith(" " * 14 + "count:"))
except (OSError, ValueError, IndexError):
    real = None
if real is not None:
    got = parked()                                     # the lists move between the two reads: a few pages
    check("the real /proc/zoneinfo: the sum of its count: lines", got is not None and abs(got - real * PAGE) < 64 * 1024**2,
          f"{got} != {real * PAGE}")

print("FAILED: " + ", ".join(failures) if failures else "all checks passed")
sys.exit(1 if failures else 0)
