# Jetson Orin Nano — power & clock profiles

The Orin Nano exposes power budgets via `nvpmodel` and clock locking via
`jetson_clocks`. The anomaly detector is tiny (a 10→64→32→6 MLP), so inference
fits comfortably in the 7 W envelope; raise to 15 W (MAXN) only if you co-locate
the TPM synchronisation loop and bifurcation simulation on the same device.

## List / set power mode
```bash
sudo nvpmodel -q                 # query current mode
sudo nvpmodel -m 1               # 7W  (id 1 on Orin Nano)
sudo nvpmodel -m 0               # 15W / MAXN (id 0)
```

## Lock clocks to the current mode's maximum (reduces latency jitter)
```bash
sudo jetson_clocks               # pin CPU/GPU/EMC clocks high
sudo jetson_clocks --show        # verify
```

## Recommended operating points
| Workload                                   | nvpmodel | jetson_clocks | Rationale                        |
|--------------------------------------------|----------|---------------|----------------------------------|
| Detector inference only                    | 7W (1)   | optional      | MLP is sub-millisecond           |
| Detector + live TPM sync + bifurcation sim | 15W (0)  | recommended   | CPU-bound sync benefits from MAXN|
| Field / battery deployment                 | 7W (1)   | off           | maximise endurance               |

## Measuring power & thermals
```bash
sudo tegrastats --interval 1000  # POM_5V_IN gives module power (mW)
```
The Prometheus exporter (`deployment/monitoring/exporter.py`) reads `tegrastats`
best-effort and publishes `pqedge_node_power_mw` / `pqedge_node_temp_c`.
