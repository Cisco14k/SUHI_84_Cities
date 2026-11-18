"""Driver: run hr.py for the subset when each city's 2016-2020 trajectories (overpass winds) exist; Landsat years outside 2016-2020 have no wind."""
import os, sys, time, glob, subprocess
ROOT = "/home/claude/n84"
todo = [int(x) for x in sys.argv[1].split(",")]
t0 = time.time()
while todo:
    ready = [i for i in todo if os.path.exists(f"{ROOT}/data/ls/{i:02d}.npz") and
             (all(os.path.exists(f"{ROOT}/data/traj/{i:02d}_{y}.npz") for y in (2016, 2017, 2020)) or time.time() - t0 > float(sys.argv[2]))]
    if not ready:
        time.sleep(60); continue
    i = ready[0]; todo.remove(i)
    subprocess.run(["python3", f"{ROOT}/src/hr.py", str(i)], env=dict(os.environ, NREC="6000", MAXSC="10"))
