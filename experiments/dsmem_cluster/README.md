# DSMEM Cluster Profiling on B200/B300

This experiment reproduces a Figure-5-style characterization for Blackwell GPUs:

1. SM-to-SM DSMEM access latency vs. thread-block cluster size.
2. DSMEM aggregate communication bandwidth vs. cluster size.
3. Estimated active SM count constrained by cluster residency.
4. First-wave `%smid` residency sanity check for each cluster size.

The benchmark is intended for B200 (`sm_100a`) and B300 (`sm_103a`) servers. It emits JSON results and a plotting script produces the three-panel figure.

## Files

| File | Purpose |
| --- | --- |
| `dsmem_cluster_bench.cu` | CUDA microbenchmark for latency, bandwidth, local-SMEM sanity, SMID residency, and active-SM occupancy. |
| `Makefile` | Builds the benchmark binary. |
| `run_dsmem_cluster.py` | Builds/runs the benchmark and stores JSON results. |
| `plot_dsmem_cluster.py` | Plots Figure-5-style latency/bandwidth/active-SM panels. |
| `results/` | Suggested output directory for raw JSON results. |
| `figures/` | Suggested output directory for generated plots. |

## What Is Measured

### DSMEM latency

Each cluster contains `C` CTAs. CTA rank 0 performs a dependent pointer-chase through peer CTA distributed shared memory using `cluster.map_shared_rank()`. For `cluster_size=C`, the benchmark sweeps `peer_delta=1..C-1` and reports both per-peer rows and the mean of peer medians. The dependency chain prevents memory-level parallelism from hiding latency. For `cluster_size=1`, the same kernel acts as a local shared-memory baseline.

The global-memory baseline uses the same dependent pointer-chase pattern over a large global-memory array. Use a large enough `--global-latency-elems` to exceed cache capacity.

### DSMEM bandwidth

Every CTA initializes a per-CTA shared-memory buffer and then reads vectorized `uint4` data from a peer CTA in the same cluster. The reported bandwidth is aggregate remote-read traffic:

```text
bandwidth = grid_blocks * bandwidth_iters * bandwidth_bytes / elapsed_time
```

The global-memory baseline is a STREAM-like vectorized read kernel over a large global-memory buffer.

The benchmark also emits `local_smem_bandwidth`, which is a local shared-memory bandwidth sanity check. The plot draws this as a dotted line and global memory as a dashed line.

### Active SM

The benchmark uses `cudaOccupancyMaxActiveClusters()` with the same cluster dimension, block size, and dynamic shared-memory usage as the bandwidth kernel. It reports:

```text
active_sms_estimate = active_clusters_estimate * cluster_size
```

This is the right value for the paper-style right panel because thread-block clusters can reduce residency even when the GPU has more physical SMs.

The benchmark also launches a tiny `%smid` probe at the same cluster size and reports the unique SM IDs touched by the first grid wave as `smid_residency`. This is a sanity check for the occupancy estimate, not a replacement for the occupancy API.

## Build

```bash
cd experiments/dsmem_cluster
make ARCH=sm_100a   # B200
make ARCH=sm_103a   # B300
```

The default `ARCH` is `sm_100a`.

## Run on B200

```bash
cd experiments/dsmem_cluster
python3 run_dsmem_cluster.py \
  --gpu-label B200 \
  --arch sm_100a \
  --cluster-sizes 1,2,4,8,16 \
  --output results/b200.json
```

## Run on B300

```bash
cd experiments/dsmem_cluster
python3 run_dsmem_cluster.py \
  --gpu-label B300 \
  --arch sm_103a \
  --cluster-sizes 1,2,4,8,16 \
  --output results/b300.json
```

## Quick Smoke Test

Use this first after logging into a GPU server:

```bash
cd experiments/dsmem_cluster
python3 run_dsmem_cluster.py --gpu-label B200 --arch sm_100a --quick --output results/b200_quick.json
```

`--quick` reduces repeat counts, buffer sizes, and cluster sizes to `1,2,4`. Do not use quick-mode results for the paper figure.

## Plot

Single GPU:

```bash
cd experiments/dsmem_cluster
python3 plot_dsmem_cluster.py results/b200.json --output figures/b200_dsmem_cluster.png
```

B200 and B300 together:

```bash
cd experiments/dsmem_cluster
python3 plot_dsmem_cluster.py \
  results/b200.json results/b300.json \
  --output figures/b200_b300_dsmem_cluster.png
```

Install plotting dependency if needed:

```bash
python3 -m pip install matplotlib
```

## Recommended Paper-Quality Settings

The defaults are already conservative enough for stable numbers:

```text
--repeats 30
--warmup 5
--latency-iters 200000
--bandwidth-iters 4096
--latency-elems 4096
--bandwidth-bytes 32768
--global-latency-elems 67108864
--global-bandwidth-bytes 536870912
```

For final results, run on an otherwise idle GPU, disable MIG, and prefer fixed application clocks. If `cluster_size=16` is unsupported by a driver/GPU configuration, the JSON row is marked `supported=false` and the plot omits that bar.

## Output Format

`run_dsmem_cluster.py` writes a JSON file with a `rows` array. Important row types:

- `metadata`: GPU name, SM count, max cluster-size attribute, SM clock attribute.
- `global_latency`: median global dependent-load latency in cycles.
- `global_bandwidth`: median global-memory bandwidth in TB/s.
- `local_smem_bandwidth`: median local shared-memory bandwidth sanity baseline in TB/s.
- `dsmem_latency`: median DSMEM dependent-load latency in cycles for each cluster size.
- `dsmem_latency_peer`: per-peer DSMEM dependent-load latency for each `peer_delta`.
- `dsmem_bandwidth`: median DSMEM aggregate bandwidth in TB/s for each cluster size.
- `active_sm`: estimated active clusters and active SMs for each cluster size.
- `smid_residency`: unique `%smid` values observed in a first-wave residency probe.

## Notes on B200/B300 Cluster Sizes

- CUDA's portable cluster size is 8 CTAs per cluster.
- B200 is documented by NVIDIA as supporting a non-portable cluster size of 16 when `cudaFuncAttributeNonPortableClusterSizeAllowed` is enabled.
- B300/GB300 is compute capability 10.3; public documentation does not currently give the same explicit 16-CTA statement, so this benchmark treats `cluster_size=16` empirically. If the driver rejects it, the row is emitted as `supported=false` and omitted from plots.
