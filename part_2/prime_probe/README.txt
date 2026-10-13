Prime+Probe contribution
========================

Copy this prime_probe folder into the team repository's part_2 directory.

Code
----
code/pp_demo.c       Small Prime+Probe demonstration (Task 2.1).
code/pp_measure.c    Calibration, signal and controls (Task 2.2).
code/analyze_pp.py   Validate raw samples; regenerate metrics and plots.

Manual build/run commands are given below. No compiled program needs to
be added beside the C source files. The SSH campaign runner is omitted.

Evidence and source versions
----------------------------
results/prime_probe/20261009T174120Z-700cc0:
    Seven measured hosts plus unsupported Thunderbird. Archived sources
    are the earlier one-buffer implementation with a xorshift shuffle.
results/prime_probe/20261009T180716Z-6bbdd6:
    Current simplified sources, measured on Skylark and Sunbird. This
    measurement version uses srand()/rand(); sequences depend on libc.

The current code matches only the second campaign. The earlier campaign
retains its exact source snapshots inside the host archives. Each campaign
has its own metrics; cite the corresponding campaign/source version.

Each host archive contains raw samples, events, CPU/compiler settings,
source snapshots and original hash manifests. The binaries, assembly and
build logs inside are dependencies of the existing artifact validator;
the handout does not independently require those compiled artifacts.
The archive format avoids including a second copy of extracted data.

Replay saved results
--------------------
Run these commands from this prime_probe directory (Linux):

    python -m pip install numpy matplotlib
    for c in results/prime_probe/*; do
        for a in "$c"/*.tar.gz; do
            h="$(basename "$a" .tar.gz)"
            mkdir -p "$c/$h"
            tar -xzf "$a" -C "$c/$h"
        done
        python code/analyze_pp.py "$c"
    done

For Windows, extract each HOST.tar.gz into its campaign's HOST directory,
then run python code/analyze_pp.py results/prime_probe/CAMPAIGN.
The included summary.csv and per_run.csv allow immediate inspection.
Plots (distributions, Prime+Probe traces and run variation) are regenerated
by the analyzer; duplicate PNG/PDF copies are not needed in this handoff.

Build and run manually (Linux/x86)
---------------------------------
    gcc -O2 -std=c11 -Wall -Wextra -Wpedantic -Werror code/pp_demo.c -o pp_demo
    ./pp_demo 0
    ./pp_demo 1
    gcc -O2 -std=c11 -Wall -Wextra -Wpedantic -Werror code/pp_measure.c -o pp_measure
    ./pp_measure CPU WAYS SETS LINE_BYTES 5922026 > raw.csv 2> events.jsonl

Replace the geometry/CPU arguments with the host's live allowed CPU and
L1 data-cache geometry. The demo defaults are Skylark's 8 ways, 64 sets,
64-byte lines; override -DWAYS, -DLINE_SIZE and -DSTRIDE when building it
for another geometry. pp_measure defaults to 10,000 calibration samples
per bit and ten balanced 10,000-trial batches per condition. It pins to the
selected CPU and records its threshold before evaluation. TSC ticks are
not assumed to equal core cycles. This is a same-process, same-core test.

For new measurements, run pp_measure on each supported assigned host and
retain raw.csv, events.jsonl, the source, compiler flags, command and machine
settings. Record unsupported cases separately. The saved-result replay
commands above validate the original archived campaigns. ARM and unsupported
address mappings are recorded as unsupported; Skylark's weak signal and
all controls must remain visible.

Before final upload
-------------------
Task 2.1 (handout page 2) asks comments for every variable and nonblank line,
including assembly. Current code has selective comments; finish the
required annotations before the final submission, keeping these executed
snapshots with the measurements. The team report must identify source
versions, contributions and the submitted repository commit.

The team separately supplies its final report/LaTeX, slides/source, video,
coverage checklist, references and work logs. Do not copy blank templates,
baseline review archives or draft reports into this Prime+Probe handoff.

Include diagnostics only when the final explanation uses their results:
then add the corresponding diagnostic source, raw data/settings and used
figure. The October 9 probe-sensitivity check is preserved in the original
Project2/results/prime_probe_diagnostics directory; it is not part of this
minimal handoff. The old October 6 PDF describes a different campaign and
is not a report of these submitted measurements.
