# ToN-IoT sample-level federated unlearning

Đã triển khai và chạy pilot không DT: leakage-aware preprocessing → 5-client FedAvg → original/no-op/retain-only/fresh retrain → ConPFU-inspired GA/GD/KD → retained-only recovery → utility, prediction similarity và MIA controls.

**Pilot chạy thành công về kỹ thuật nhưng chưa đạt gate kiểm chứng FU. DT chưa triển khai.** MIA retained control yếu và retain-only gần như tương đương FU. Xem [báo cáo](docs/validation_report.md), [protocol](docs/experiment_protocol.md), [audit ban đầu](docs/implementation_audit.md), [kế hoạch](docs/implementation_plan.md).

## Chạy lại

PowerShell từ `E:/budget-aware-code`, dùng `python` tại `D:/python/python.exe` (torch CPU đã có). Manifest lưu phiên bản thực tế; requirements không phải lockfile đầy đủ.

```powershell
python -m unittest discover -s tests -v
python src/data/prepare.py --config configs/data_connected.json --output data/processed/reproduction_seed42
python scripts/run_pilot.py --data data/processed/reproduction_seed42 --output runs/reproduction_smoke --smoke
python scripts/run_pilot.py --data data/processed/reproduction_seed42 --output runs/reproduction_pilot
python scripts/verify_run.py --data data/processed/reproduction_seed42 --run runs/reproduction_pilot
python scripts/plot_results.py --run runs/reproduction_pilot --output results/figures/reproduction_pilot
```

Output run/preprocessing phải chưa tồn tại. Runner từ chối data audit chưa đạt hoặc thiếu kiểm tra encoded overlap. Smoke dùng subset khai báo rõ, không làm evidence FU. Config thay đổi phải dùng run mới.

## Thành phần

- `src/data/prepare.py`: connected grouping, train-only transforms, audit.
- `src/data/federated.py`: train-only Dirichlet, nested requests, retained partitions.
- `src/fl/fedavg_core.py`: MLP, logical worker, local training và shared weighted FedAvg.
- `src/fu/conpfu_inspired.py`: alternating GA/GD, regularization/clipping, retain-only KD.
- `src/eval/metrics.py`: utility, prediction similarity, fixed calibrated loss MIA.
- `scripts/run_pilot.py`: fresh retrain, recovery, checkpoints/logs, evaluation và gate.
- `scripts/verify_run.py`: artifact hashes, ownership, nested requests, retained counts, no-op.

## Artifacts thực tế

Data hiện tại: `data/processed/connected_v3_seed42`. Pilot: `runs/pilot_v3_seed42`. Figures: `results/figures/pilot_v3_seed42`. Checkpoints, initial model, predictions, evaluation IDs, request IDs, manifests, losses, timings, coverage và gate đều đã lưu.

Các data/run v1/v2 và `smoke_seed42` cũ chỉ là diagnostics trước sửa float32 overlap; không dùng làm evidence. Raw CSV bất biến. Raw/mapping/request IDs chỉ thuộc offline research/local workers, không publish công khai hoặc gửi server/DT.

Đây là mô phỏng sequential CPU trong một process, chưa phải dịch vụ phân tán. Bytes là tensor payload, không phải wire bytes; chưa đo peak memory và chi phí mạng end-to-end. Chưa có DP, deletion guarantee, multi-seed validation hoặc full DT.
