#!/usr/bin/env python3
"""Checks for patch 0119 (the startup budget counts the pages parked on the kernel's per-CPU free lists beyond their
resting level), run inside the image that scripts/prepare.sh built (CPU only: no GPU, no model, about a second):

    docker run --rm --entrypoint python -v "$PWD/tools/parked_pages_check.py:/c.py" tensorfold-glm53:v0.6.0 /c.py

or on any machine against TensorFold's patched source (no torch needed):

    PYTHONPATH=/path/to/TensorFold/src python tools/parked_pages_check.py

Synthetic /proc/zoneinfo and /proc/meminfo files (capacity.ZONEINFO / MEMINFO point at them) and a stand-in for torch:
page lists on several CPUs, zones and nodes (only the ``count:`` lines under ``pagesets`` are summed, less their
``high_min:`` lines: the excess over the resting level), a warm start's lists (counts ~0), lists without high_min
(the first 512 MiB are resting), zones without lists, no lists at all, an unreadable or garbled file (0, never an
error), and the figures the startup budget reads (``available_bytes`` / ``page_room`` given ``parked``, on a unified and
on a discrete GPU), which a live caller that passes none does not see. Exit code 1 when a check fails, and on an
image without the patch.
"""
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

from tensorfold.cuda import capacity

GIB, KIB = 1024**3, 1024
PAGE = os.sysconf("SC_PAGE_SIZE")
HIGH_MIN = 2146                          # pages a CPU's list holds at rest, as the kernel prints it on a Spark
failures = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f" ({detail})" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def cpus(counts, high_min=HIGH_MIN, indent=4):
    """The page lists of one zone: a ``cpu:`` block a count, as the kernel prints them (no high_min line when None)."""
    pad = " " * indent
    rest = "" if high_min is None else f"{pad}          high_min: {high_min}\n"
    return "".join(f"{pad}cpu: {n}\n{pad}          count:    {c}\n{pad}          high:     5000\n{pad}          batch:    63\n"
                   f"{rest}{pad}          high_max: 198283\n  vm stats threshold: 110\n"
                   for n, c in enumerate(counts))


def zone(node, name, counts=None, managed=1000, high_min=HIGH_MIN):
    """One zone's block: stats lines, the page lists when ``counts`` is given, start_pfn."""
    text = f"Node {node}, zone {name:>8}\n  pages free     10\n        managed  {managed}\n      nr_free_cma  0\n"
    if counts is not None:
        text += "  pagesets\n" + cpus(counts, high_min)
    return text + "  start_pfn:           4096\n"


def parked():
    return capacity.parked_bytes() if hasattr(capacity, "parked_bytes") else None      # an image without the patch


def run(zoneinfo, meminfo, torch=None, zone_path=None):
    """The figures the startup budget reads with those files in place of /proc's: ``avail`` and ``room`` get the parked
    bytes as admit() passes them; ``live`` is available_bytes as the live gates call it, with none."""
    d = Path(tempdir.name)
    if zoneinfo is not None:
        (d / "zoneinfo").write_text(zoneinfo)
    (d / "meminfo").write_text(meminfo)
    capacity.ZONEINFO = zone_path or str(d / "zoneinfo")
    capacity.MEMINFO = str(d / "meminfo")
    p = parked()
    try:
        return NS(parked=p, info=capacity._meminfo(p or 0), plain=capacity._meminfo(),
                  avail=capacity.available_bytes(torch, p or 0) if torch else None,
                  room=capacity.page_room(torch, p or 0) if torch else None,
                  live=capacity.available_bytes(torch) if torch else None)
    except TypeError:                                                  # an image without the patch: no such arguments
        plain, live = capacity._meminfo(), capacity.available_bytes(torch) if torch else None
        return NS(parked=p, info=plain, plain=plain, avail=live, room=capacity.page_room(torch) if torch else None, live=live)


def fake(integrated, free=0, total=0):
    props = NS(is_integrated=integrated)
    return NS(cuda=NS(mem_get_info=lambda: (free, total), get_device_properties=lambda i: props))


def meminfo(total_gib, avail_gib):
    return (f"MemTotal:       {total_gib * GIB // KIB} kB\nMemFree:        {GIB // KIB} kB\n"
            f"MemAvailable:   {int(avail_gib * GIB) // KIB} kB\nBuffers:        1 kB\n")


tempdir = tempfile.TemporaryDirectory()
os.environ.pop("TENSORFOLD_MEMORY_RESERVE_GIB", None)
check("capacity.parked_bytes exists (patch 0119 applied)", hasattr(capacity, "parked_bytes"))

MEM = meminfo(128, 100)
unified = fake(True, free=10 * GIB, total=128 * GIB)
discrete = fake(False, free=60 * GIB, total=80 * GIB)
reserve = capacity.reserve_bytes(128 * GIB, host=True)             # a tenth of the host's memory (12.8 GiB)

# several CPUs on several zones of two nodes: every count less every high_min, nothing else
counts_a, counts_b, counts_c = [200000, 3143, 2332, 17], [100000, 100000, 100000], [400000]
text = (zone(0, "DMA", counts_a) + zone(0, "DMA32", None) + zone(0, "Normal", counts_b) + zone(1, "Normal", counts_c)
        + zone(1, "Movable", None))
n = len(counts_a + counts_b + counts_c)
excess = (sum(counts_a + counts_b + counts_c) - n * HIGH_MIN) * PAGE
r = run(text, MEM, unified)
check("sums every CPU, zone and node, less each list's resting level", r.parked == excess, f"{r.parked} != {excess}")
check("MemAvailable includes the excess", r.info["MemAvailable"] == 100 * GIB + excess)
check("MemTotal is the file's", r.info["MemTotal"] == 128 * GIB)
check("available_bytes (unified GPU) is MemAvailable + excess - the 10 % reserve",
      r.avail == 100 * GIB + excess - reserve, f"{r.avail}")
check("page_room (unified GPU) is MemAvailable + excess", r.room == 100 * GIB + excess)
check("a live caller (no parked argument) keeps MemAvailable", r.live == 100 * GIB - reserve and
      r.plain["MemAvailable"] == 100 * GIB)
r = run(text, MEM, discrete)
check("a discrete GPU is bounded by the smaller of both: here its own free memory",
      r.avail == 60 * GIB - capacity.reserve_bytes(80 * GIB), f"{r.avail}")
check("page_room is None on a discrete GPU", r.room is None)

# GiBs of them, like after an image copy: 6 GiB over the resting level is 6 GiB more budget
six = 6 * GIB // PAGE
r = run(zone(0, "Normal", [six // 4 + HIGH_MIN] * 4), meminfo(128, 90), unified)
check("6 GiB parked over the resting level raise the budget by 6 GiB",
      r.avail - (90 * GIB - reserve) == six // 4 * 4 * PAGE == 6 * GIB, f"{r.avail}")

# a warm start: the lists hold their resting level and a little more, or less
for label, counts in (("at their resting level", [HIGH_MIN] * 20), ("a little over it (0.11 GiB)", [HIGH_MIN + 1440] * 20),
                      ("below it", [10, 0, 5, HIGH_MIN - 1] * 5), ("empty", [0] * 20)):
    r = run(zone(0, "Normal", counts), MEM, unified)
    small = r.parked is not None and r.parked < GIB // 4
    check(f"a warm start, lists {label}: under the log line's 0.25 GiB, MemAvailable within 0.25 GiB",
          small and r.info is not None and r.info["MemAvailable"] - 100 * GIB < GIB // 4, f"{r.parked}")
check("a warm start with the lists below their resting level counts nothing",
      run(zone(0, "Normal", [100] * 20), MEM).parked == 0)

# no high_min lines (kernels before 6.7): the first 512 MiB are resting
half = capacity.PARKED_REST if hasattr(capacity, "PARKED_REST") else None
r = run(zone(0, "Normal", [(GIB + GIB // 2) // PAGE // 2] * 2, high_min=None), MEM, unified)
check("no high_min: the first 512 MiB are resting", half == GIB // 2 and r.parked == GIB, f"{r.parked}")
check("no high_min, lists under 512 MiB: nothing", run(zone(0, "Normal", [1000] * 4, high_min=None), MEM).parked == 0)

# other lines named count, and the lines after the lists, are not page lists
noise = (zone(0, "Normal", [HIGH_MIN + 10]).replace("  start_pfn:           4096\n",
                                                    "  start_pfn:           4096\n      count:    99999\n      high_min: 7\n")
         + "Node 0, zone Other\n      count:    77777\n  pagesets\n    cpu: 0\n              high:     1\n")
r = run("  count: 5\n      count: 6\n" + noise, MEM)
check("only the count: and high_min: lines under pagesets", r.parked == 10 * PAGE, f"{r.parked}")

# the kernel's other layouts of the lists
r = run("Node 0, zone Normal\n  pagesets\n    cpu: 0\n              count: 2200\n              high: 9\n"
        "              high_min: 2000\n  vm stats threshold: 8\n    cpu: 1\n              count: 2100\n"
        "              high_min: 2000\n  all_unreclaimable: 0\n", MEM)
check("a list that ends at the next field", r.parked == 300 * PAGE, f"{r.parked}")

# no page lists, an empty, an unreadable and a garbled file: nothing parked, the start goes on
no_lists = zone(0, "DMA", None) + zone(0, "Normal", None)
bad = zone(0, "Normal", [10 ** 6])
for label, text_, path in (("no page lists", no_lists, None), ("an empty file", "", None),
                           ("a missing file", None, str(Path(tempdir.name) / "none")),
                           ("a directory", None, tempdir.name),
                           ("a count that is not a number", bad.replace("count:    1000000", "count:    x"), None),
                           ("a count with no value", bad.replace("count:    1000000", "count:"), None),
                           ("a high_min that is not a number", bad.replace("high_min: 2146", "high_min: x"), None),
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

# this host's real file, when it has page lists in a layout this can read (any indentation): the answer is a figure
# between nothing and every parked page, and near an independent reading of the same file (the lists move between the
# two reads; a GiB is far more than they move, and far less than a wrong parse)
capacity.ZONEINFO, capacity.MEMINFO = "/proc/zoneinfo", "/proc/meminfo"
try:
    real = Path("/proc/zoneinfo").read_text()
except OSError:
    real = ""
counts = [int(v) for v in re.findall(r"^\s+count:\s+(\d+)\s*$", real, re.M)]
rests = [int(v) for v in re.findall(r"^\s+high_min:\s+(\d+)\s*$", real, re.M)]
if "pagesets" in real and counts and rests:
    want, got = max(0, sum(counts) - sum(rests)) * PAGE, parked()
    check("the real /proc/zoneinfo: between nothing and every parked page, near the count: less high_min: lines",
          got is not None and 0 <= got <= sum(counts) * PAGE and abs(got - want) < GIB, f"{got} vs {want}")
else:
    print("skip the real /proc/zoneinfo (no page lists with count: and high_min: lines here)")

print("FAILED: " + ", ".join(failures) if failures else "all checks passed")
sys.exit(1 if failures else 0)
