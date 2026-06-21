# Deployment — Jetson Orin Nano + 5G modem

> **Maturity:** the simulation and experiments in this repository are validated
> on CPU. The Jetson container, the TensorRT engine build, and the modem bring-up
> below are a **documented prototype path** — they are written to be correct and
> runnable on the target, but they have **not** been validated on hardware inside
> this repository's CI. Treat them as an integration guide, not a benchmarked
> result. See the paper's *Limitations* section.

## 1. Components

| Path                              | Purpose                                              |
|-----------------------------------|------------------------------------------------------|
| `Dockerfile.jetson`               | L4T/JetPack image linking CUDA + TensorRT            |
| `docker-compose.yml` (repo root-rel) | detector exporter + Prometheus + Grafana          |
| `tensorrt/convert_to_tensorrt.py` | PyTorch → ONNX → TensorRT engine                     |
| `tensorrt/serve_infer.py`         | inference server (TensorRT→ONNXRuntime→PyTorch)      |
| `monitoring/exporter.py`          | Prometheus exporter (detector + node power/thermals) |
| `power/`                          | `nvpmodel` / `jetson_clocks` profiles                |

## 2. Bring-up on the device

```bash
# 0) Flash JetPack 6 (L4T r36) and boot the Orin Nano.
# 1) Build the image ON the device (arm64):
sudo docker build -f deployment/Dockerfile.jetson -t pqedge:jetson .
# NOTE: install the JetPack-matched PyTorch wheel inside the image — see the
#       commented URL in Dockerfile.jetson (the x86 PyPI wheel will not work).

# 2) Set a power mode and lock clocks:
sudo bash deployment/power/set_power_mode.sh 7w

# 3) Build the accelerated detector engine:
python3 deployment/tensorrt/convert_to_tensorrt.py \
    --dataset datasets/telemetry.csv \
    --onnx deployment/tensorrt/detector.onnx \
    --engine deployment/tensorrt/detector.plan
# If tensorrt python is present this writes detector.plan; otherwise it writes
# the ONNX and prints the trtexec command to finish the build.

# 4) Serve inference (auto-selects TensorRT → ONNXRuntime → PyTorch):
python3 deployment/tensorrt/serve_infer.py
```

## 3. Monitoring stack

```bash
# From the repo root (CPU image is fine for the exporter):
docker compose -f deployment/docker-compose.yml up --build
#  exporter  -> http://localhost:9108/metrics
#  Prometheus-> http://localhost:9090
#  Grafana   -> http://localhost:3000  (anonymous viewer enabled; import
#               deployment/monitoring/grafana_dashboard.json)
```

Exported series: `pqedge_detector_inference_seconds`, `pqedge_detector_class_score{class=...}`,
`pqedge_detector_alerts_total`, and (on-device) `pqedge_node_power_mw` / `pqedge_node_temp_c`.

## 4. 5G modem integration (multi-carrier bifurcation transport)

The (2,2) split bearer sends **share 1** and **share 2** over two independent
bearers. Two reference M.2 modems:

- **Quectel RM520N-GL** (5G Sub-6 / mmWave, M.2 B-key, exposes a QMI/MBIM WWAN
  interface; RGMII/USB3 to the host).
- **SIMCom SIM8202G-M2** (5G NSA/SA, M.2, PCIe/USB3; QMI via the Qualcomm stack).

### 4.1 Physical separation of the two shares
True path diversity (the security premise of bifurcation) requires the two
shares to traverse **different** radio paths. Options, strongest first:

1. **Two modems / two operators** — one share per modem, each on a different
   carrier (ideally different RAN + core). Best path independence.
2. **One modem, two PDU sessions / DNNs** — separate APNs routed to different
   UPFs. Logically distinct bearers; shares correlation-bounded by the shared RF
   front-end (document this as residual correlation).
3. **One modem, two component carriers (CA)** — e.g. n78 + n258. *Lowest*
   independence: a single RF failure drops both shares. Use only as a fallback.

> Honesty note: options 2–3 do **not** give information-theoretic path
> independence; a compromise of the shared modem/PDU anchor sees both shares.
> The confidentiality argument is strongest under option 1.

### 4.2 Bring up two WWAN interfaces (QMI example)
```bash
# Identify the two control nodes (one per modem):
ls /dev/cdc-wdm*           # e.g. /dev/cdc-wdm0  /dev/cdc-wdm1

# Raw-IP mode + connect each modem to a different APN/operator:
sudo qmicli -d /dev/cdc-wdm0 --wda-set-data-format=raw-ip
sudo ip link set wwan0 down && echo Y | sudo tee /sys/class/net/wwan0/qmi/raw_ip
sudo qmicli -d /dev/cdc-wdm0 \
  --wds-start-network="apn='carrierA.apn',ip-type=4" --client-no-release-cid

sudo qmicli -d /dev/cdc-wdm1 --wda-set-data-format=raw-ip
sudo ip link set wwan1 down && echo Y | sudo tee /sys/class/net/wwan1/qmi/raw_ip
sudo qmicli -d /dev/cdc-wdm1 \
  --wds-start-network="apn='carrierB.apn',ip-type=4" --client-no-release-cid

sudo udhcpc -i wwan0 ; sudo udhcpc -i wwan1
```

### 4.3 Pin each share to a carrier with policy routing
```bash
# Two tables: share 1 -> wwan0, share 2 -> wwan1.
sudo ip route add default dev wwan0 table 101
sudo ip route add default dev wwan1 table 102
# Mark per-share flows (the transport layer sets SO_MARK 0x1 / 0x2 per share):
sudo ip rule add fwmark 0x1 table 101
sudo ip rule add fwmark 0x2 table 102
```

The behavioural model in `bifurcation/carriers.py` mirrors this split-bearer
(two carriers, both shares required, parallel-leg latency = max of the two
legs). Replacing the simulated `CarrierModel`s with live socket sends bound to
`wwan0`/`wwan1` (via `SO_BINDTODEVICE` + `SO_MARK`) turns the simulation into a
hardware transport with no change to the reconstruction logic.

### 4.4 Useful AT / diagnostics
```bash
# Signal & registration (either modem, over its AT port, e.g. /dev/ttyUSB2):
echo -e 'AT+QENG="servingcell"\r' > /dev/ttyUSB2   # Quectel serving cell
echo -e 'AT+CPSI?\r' > /dev/ttyUSB2                # SIMCom serving cell info
echo -e 'AT+CSQ\r'   > /dev/ttyUSB2                # signal quality
```
