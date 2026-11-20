# GB300 spilling-sweep figure

The [paper figure](gb300-spill-sweeps.pdf) compares the saved host-memory and
device-memory limit sweeps at SF1000, SF3000, and SF10000. The PDF and
[SVG](gb300-spill-sweeps.svg) are vector exports with editable text; the
[PNG](gb300-spill-sweeps.png) is 600 dpi at 11 inches wide in landscape format. The
[aggregated data](gb300-spill-sweeps-data.csv) and [plotting script](plot_sweeps.py)
are also saved here.

**Suggested caption:** *Spilling under host- and device-memory limits on a
GB300. Each point sums one execution of all 22 TPC-H queries in fresh Sirius
processes. Lines show total query time; bars show cumulative logical DataBatch
movement. GPU→HOST is separate from the disk-directed bar, whose segments
distinguish HOST→DISK and GPU→DISK. Movement is measured in the destination
representation and is not unique data volume or physical bus traffic. The
host-limit sweep uses the default GPU configuration and scales the initial
host pools with the cap; the device-limit sweep fixes host capacity at
384 GiB. Host limits are binary GiB, while GPU limits are decimal GB. Each
point is a single measurement, so no error bars are shown.*

Read each row as its own sweep. The host sweep did not clear the OS page
cache, while the device sweep requested that cached dataset pages be discarded;
their runtimes should not be interpreted as a controlled comparison between
rows. Within a column, both runtime axes use the same scale, while movement
axes adapt to the amount moved. The disk-directed bar stacks only the two
transitions with a disk destination. It is shown beside GPU→HOST rather than
added to it, because the transitions can count different representations of
the same batch.

The figure reads the checked-in
[host-sweep results](../gb300-spill-sweep/results/) and
[device-sweep results](../gb300-device-spill-sweep/results/). It validates that
all 22 queries succeeded at every cap and that per-query movement sums match
the saved totals. To regenerate it with Matplotlib 3.11.2 or later:

```bash
python bench/gb300-spill-figures/plot_sweeps.py
```
