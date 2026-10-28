# Hướng dẫn triển khai Digital Twin cho khóa luận

Ngày đối chiếu: 27/10/2026. Repository: https://github.com/thaodinh97/buget-aware, commit `4c158fe044ccd36706d75ae9a1257be18d1f0707`.

## 1. Chốt đúng phạm vi trước khi viết code

Digital Twin (DT) trong đề tài là **execution/resource twin cho FL/FU**: theo dõi trạng thái thực thi, dự báo chi phí, chọn kế hoạch recovery, gửi lệnh thực thi, rồi dùng số đo mới để cập nhật dự báo. Không cần mô hình 3D hay AWS IoT TwinMaker.

Hai chiều bắt buộc:

1. Hệ thực thi → twin: số mẫu retained, phân bố nhãn tổng hợp, phiên bản dữ liệu/mô hình, thời gian train, thời gian request, memory, lỗi và tình trạng online.
2. Twin → hệ thực thi: kế hoạch thật sự quyết định client nào train và mô hình nào được tổng hợp.

Docker Compose là môi trường tạo các node thực thi, **không tự nó tạo ra DT**. Năm container vẫn dùng chung kernel/tài nguyên host; đó là mô phỏng node có quota, không tương đương năm máy vật lý độc lập. EC2 giúp kiểm chứng trên các VM thực sự phân tán.

### BIPPO dùng ở đâu?

PDF BIPPO có tên “BIPPO: Budget-Aware Independent PPO for Energy-Efficient Federated Learning Services”, arXiv:2511.08142v1. Section III mô tả FL và mô hình năng lượng; Section IV mô tả IPPO và bộ lấy mẫu theo ngân sách; Algorithm 1 ở trang PDF 6. Đây không phải đặc tả Digital Twin cho Federated Unlearning.

| Nội dung | BIPPO | Khóa luận này |
|---|---|---|
| Công việc được điều phối | Train FL | Recovery sau FU |
| Ngân sách chính | Năng lượng ước lượng | Thời gian, bytes; tiền cloud là lớp riêng |
| Tín hiệu chất lượng | Accuracy trong các vòng FL | Utility, độ gần retrain, kiểm chứng FU |
| Dữ liệu thực nghiệm | Fashion-MNIST/CIFAR-10 | ToN-IoT Network |
| Planner khởi đầu | IPPO + sampler | Duyệt subset với K=5 |

Có thể học ý tưởng tách policy đề xuất và bước kiểm tra ngân sách. Không sao chép công thức cộng năng lượng thành công thức cộng latency: train song song chịu ảnh hưởng client chậm nhất và các tài nguyên dùng chung. Không dùng test accuracy để chọn kế hoạch trong khóa luận; chỉ dùng validation/calibration được quy định trước.

### Tình trạng code đang có

- `src/fl/fedavg_core.py`: `ClientWorker.train`, `aggregate`, `fedavg`; hàm `fedavg` đang gọi worker tuần tự trong một process.
- `src/fu/conpfu_inspired.py`: `alternating` và `distill` cho bước FU cục bộ.
- `scripts/run_pilot.py`: original/retrain/retain-only/FU và checkpoint, chưa có transport mạng.
- `scripts/verify_run.py`: kiểm chứng artifact/run.
- `configs/pilot.json`: 5 client, hidden `[64,32]`, local epoch 1, batch 512, recovery 2 vòng.
- `docs/validation_report.md`: pilot chưa chứng minh lợi ích FU riêng; MIA control yếu; `dt_allowed=false`.

Bạn đã yêu cầu tiếp tục thiết kế DT, nên bộ này cho phép làm thí nghiệm hạ tầng ngay, đồng thời lưu trạng thái `infrastructure_only`. Không sửa gate cũ thành true. Kết luận về học quên vẫn cần evidence FU; DT không giải quyết thay vấn đề đó.

## 2. Kiến trúc cần triển khai

```text
Offline research/provisioner (máy nghiên cứu riêng)
  ├─ kiểm tra preprocessing, ownership, request, checkpoint
  ├─ tạo retained shard cho từng client
  └─ tạo checkpoint sau FU + metadata tổng hợp cho central

Central container / EC2-0
  ├─ Coordinator: barrier, version check, FedAvg, checkpoints
  ├─ State Twin: snapshot, timestamp, data version
  ├─ Cost Predictor: dự báo và cập nhật bằng feedback
  ├─ Planner: thử 16 subset chứa target
  └─ Run reports: dự báo, thực tế, sai số, vi phạm budget
          │ HTTP model + lệnh / telemetry tổng hợp
  ┌───────┼────────┬────────┬────────┐
client0 client1 client2 client3 client4
  └─ mỗi node chỉ đọc shard retained của chính nó
```

Bản đầu có **6 container: 1 central + 5 client**. DT là các module trong central, vẫn là một thành phần logic riêng. Khi cần nghiên cứu overhead qua API, có thể tách DT thành container thứ bảy; chưa cần thêm một máy EC2 chỉ để chứa JSON state.

Không mount toàn bộ `data/` hoặc toàn bộ repository vào mọi worker. Central không có `X`, `y`, row IDs, F hay mapping dữ liệu. Central được nhận counts/histogram tổng hợp trong phạm vi thí nghiệm này; chúng vẫn là metadata nhạy cảm và cần kiểm soát truy cập.

## 3. Những gì bộ code kèm theo thực sự làm

| File | Vai trò |
|---|---|
| `dt/export_case.py` | Từ full pilot đã có, xuất shard retained riêng và checkpoint sau FU |
| `dt/worker.py` | FastAPI `/health`, `/state`, `/train`, reuse `ClientWorker.train` |
| `dt/codec.py` | Trao đổi tensor bằng NPZ, `allow_pickle=False`, hash phiên bản |
| `dt/planner.py` | Duyệt subset, kiểm tra time/bytes/JS/coverage/TTL |
| `dt/coordinator.py` | Calibration, train song song, barrier, weighted FedAvg, EWMA, report |
| `Dockerfile.dt` | Image Linux dùng chung cho worker và central |
| `compose.dt.yaml` | 5 worker độc lập; central chạy tác vụ theo CLI |
| `compose.ec2-worker.yaml` | 1 worker trên mỗi EC2, bind private IPv4 |
| `tests/test_dt.py` | HTTP integration, parity với FedAvg cũ, idempotency, planner và CLI |

**Đây là starter cho phase DT/recovery, không phải dịch vụ UaaS hoàn chỉnh.** Nó dùng checkpoint `after_local` hoặc `after_kd` đã sinh bởi pilot để cố định bước FU, rồi nghiên cứu tác động của recovery selection. Nó không cung cấp endpoint nhận xóa dữ liệu mới, hàng đợi bền vững, distributed lease hoặc giao dịch xóa dữ liệu thật. Phần 15 chỉ rõ cách thêm các chức năng đó.

Chiều feedback đã có: `/train` trả telemetry → coordinator cập nhật twin → lần lập kế hoạch kế tiếp đọc twin mới → chọn node khác nếu chi phí thay đổi. Trong một yêu cầu, tập client và số vòng được giữ cố định để dễ so sánh công bằng. Không gọi đây là controller đổi subset sau mỗi vòng.

## 4. Bước 1 — Chuẩn bị repository và môi trường

Các lệnh trong phần Docker local dưới đây chạy bằng **PowerShell ở thư mục gốc repository**. Chạy Linux containers bằng Docker Desktop/WSL2; không dùng Windows containers.

```powershell
git clone https://github.com/thaodinh97/buget-aware.git
Set-Location buget-aware
git checkout 4c158fe044ccd36706d75ae9a1257be18d1f0707
git switch -c codex/digital-twin
```

Giải nén bundle và chép nội dung `digital-twin-starter` vào gốc repo: thêm `dt/`, các file Docker/Compose, `requirements-dt.txt`, và `tests/test_dt.py`. Giữ nguyên `src/`, `scripts/`, `configs/` upstream. Không chép raw data vào image.

Máy chuẩn bị dữ liệu cần Python đã cài upstream dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ml.txt -r requirements-preprocessing.txt -r requirements-dt.txt
python -m unittest discover -s tests -v
docker version
docker compose version
```

Nếu bạn dùng `D:/python/python.exe` như README, có thể dùng trực tiếp môi trường đó; tránh lẫn interpreter khi chạy pip. `requirements-ml.txt` upstream chưa khóa đầy đủ dependency; lưu `pip freeze`, Python version và image digest cùng mỗi đợt thí nghiệm. Dockerfile dùng Python 3.11; cần chạy parity/regression trong Linux trước khi coi giống kết quả Windows/Python 3.13.

Chỉ chạy unit/integration không cần bộ ToN-IoT. Các bước full pilot/export phía dưới cần dữ liệu và artifact đang nằm trên máy nghiên cứu của bạn; chúng không có sẵn đầy đủ trong GitHub.

## 5. Bước 2 — Kiểm chứng full pilot và xuất dữ liệu đúng ranh giới

Nếu đã có đúng data/run v3 và cần tái sử dụng:

```powershell
python scripts/verify_run.py --data data/processed/connected_v3_seed42 --run runs/pilot_v3_seed42
```

Nếu cần tái tạo, kiểm tra đường dẫn raw trong `configs/data_connected.json` trước. Các output phải chưa tồn tại:

```powershell
python src/data/prepare.py --config configs/data_connected.json --output data/processed/dt_seed42
python scripts/run_pilot.py --data data/processed/dt_seed42 --output runs/dt_source_seed42
python scripts/verify_run.py --data data/processed/dt_seed42 --run runs/dt_source_seed42
```

Không dùng `--smoke` làm input chính cho exporter: pilot smoke chỉ dùng subset, không trùng full train ownership. Full pilot chạy xong không có nghĩa FU gate đã đạt.

Ví dụ export một yêu cầu quên 10%, từ run v3 hiện có:

```powershell
python -m dt.export_case --data data/processed/connected_v3_seed42 --run runs/pilot_v3_seed42 --fraction 0.1 --output provision
```

Output:

```text
provision/
  central/
    case.json          # config, counts/hist, target, hash; không có row IDs
    original.npz       # cho retain-only baseline
    after_local.npz    # checkpoint sau alternating FU
    after_kd.npz       # checkpoint sau KD
  client0/local.npz     # chỉ retained X/y/row_id của client0
  client1/local.npz
  client2/local.npz
  client3/local.npz
  client4/local.npz
```

Exporter giữ chính xác phân hoạch của `partitions.npz`, bỏ F khỏi toàn bộ shard triển khai, kiểm tra ownership và completeness; không phân hoạch ngẫu nhiên lại. Client target vẫn online và train trên R. Mỗi file shard có SHA-256 làm `data_version`; `/state` phải khớp `case.json`.

Bước này là **offline provisioning cho nghiên cứu**, không chứng minh raw data gốc và backup đã bị xóa. Giữ bản F phục vụ evaluator offline trong vùng tách biệt; tuyệt đối không cấp nó cho central hay recovery worker. Với yêu cầu vận hành thật, cần quy trình xóa và quản trị snapshot như phần 15.

## 6. Bước 3 — Chạy 5 client bằng Docker Compose

Tạo token ngẫu nhiên và file `.env`:

```powershell
$dtToken = python -c "import secrets; print(secrets.token_hex(32))"
"DT_TOKEN=$dtToken" | Set-Content -Encoding ascii .env
New-Item -ItemType Directory -Force runs/dt | Out-Null
docker compose -f compose.dt.yaml config --quiet
docker compose -f compose.dt.yaml build
docker compose -f compose.dt.yaml up -d --wait client0 client1 client2 client3 client4
docker compose -f compose.dt.yaml ps
docker compose -f compose.dt.yaml logs --tail 40 client4
```

Compose không publish port worker ra host: central gọi DNS nội bộ `http://client0:8000`...`http://client4:8000`. Mỗi worker có 1 CPU quota, giới hạn RAM 2 GiB, PyTorch 1 thread. Đây là cấu hình khởi đầu để đo; kiểm tra OOM và sizing thực tế bằng `docker stats`. Host/WSL2 phải đủ tài nguyên cho toàn bộ pool; nếu bị tranh chấp, đó là một scenario cần ghi lại.

Kiểm tra một worker từ trong container:

```powershell
docker compose -f compose.dt.yaml exec client0 python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
docker stats --no-stream
```

Không tăng `uvicorn --workers`: lock/cache của starter chỉ bảo vệ trong một process. Chỉ chạy một coordinator tại một thời điểm.

## 7. Bước 4 — Calibration: tạo twin bằng phép đo thật

Chạy ba vòng probe all-clients từ cùng checkpoint sau FU; model cập nhật trong probe được bỏ đi. Đây là huấn luyện thật để đo, không phải số giả:

```powershell
docker compose -f compose.dt.yaml run --rm central calibrate --case /case/case.json --initial /case/after_local.npz --twin /runs/twin_seed42.json --output /runs/calibration_seed42
```

`calibration_seed42/report.json` chứa ba vòng đo, `twin_seed42.json` chứa snapshot. Chi phí cold start/import/image build không nằm trong `actual_seconds`; phải báo deployment/setup riêng. Chi phí calibration được ghi riêng và cần tính thêm hoặc phân bổ khi báo chi phí toàn hệ thống.

Starter dùng predictor thực nghiệm tối giản:

```text
client_elapsed_i = thời gian HTTP request train đến khi đọc đủ response
client_elapsed_i gồm train, serialization và phần trao đổi ứng dụng
round_overhead = wall_time_round - max(client_elapsed_i)
T_hat = overhead_allowance + R * safety_margin * (max_i EWMA(client_elapsed_i) + EWMA(round_overhead))
```

Sau mỗi vòng thực thi:

```python
estimate_new = 0.7 * estimate_old + 0.3 * observed
```

Không diễn giải EWMA này là mô hình suy luận riêng bandwidth/CPU. Nó chỉ dùng được quanh batch/epoch/model/count/profile đã hiệu chuẩn. Thay workload, shard, URL hoặc môi trường thì tạo calibration mới. Starter hash toàn bộ `case.json` để tránh dùng nhầm twin, nhưng thay CPU quota/network profile không tự đổi case hash: bạn phải chủ động recalibrate và ghi profile vào manifest thí nghiệm.

Để tiến tới công thức chương 2.8 của thesis, thu riêng:

```text
steps_i = local_epochs * ceil(n_retained_i / batch_size)
compute_i = steps_i * tau_i
communication_i = down_bytes_i / down_Bps_i + up_bytes_i / up_Bps_i + protocol_latency_i
T_round_hat(S) = max_i(compute_i + communication_i) + T_agg + T_control
```

Đơn vị bandwidth là bytes/giây; nếu đo Mbps phải đổi `Mbps * 10^6 / 8`. Đừng cộng RTT nhiều lần khi thời gian transfer đã bao gồm round-trip đó. Với đường uplink dùng chung, cần mô hình bottleneck chung hoặc residual theo số participant; công thức max đơn giản có thể thiếu cạnh tranh mạng.

Fit `tau_i` theo bucket batch/epoch/load, dùng probe truyền payload độc lập cho up/down bandwidth; không lấy `request_elapsed - compute` rồi gọi đó là bandwidth vật lý vì nó còn chứa serialization/queue/HTTP overhead. Predictor chi tiết là bước mở rộng, chưa nằm trong starter.

## 8. Bước 5 — Lập kế hoạch theo ngân sách

Với 5 client, target luôn tham gia: có `2^(5-1) = 16` subset. Giữ `R=2` như `configs/pilot.json` trong thí nghiệm chính. Chỉ khảo sát số vòng khác trong ablation riêng.

Planner starter chọn thời gian dự báo nhỏ nhất, với điều kiện:

- `T_hat <= budget_seconds`.
- `body_bytes_hat <= budget_bytes`.
- `target ∈ S`.
- `coverage(S) >= min_coverage` với trọng số **số mẫu retained**.
- `JS(label_hist_S, label_hist_all_retained) <= max_js`.
- Telemetry của mọi client được chọn còn hạn TTL.

Mặc định `max_js=0.05` theo log tự nhiên, `min_coverage=0.2`, `margin=1.25`, TTL 600 giây, overhead allowance 0.5 giây chỉ là **giá trị khởi đầu**, không phải hằng số từ paper. Hiệu chuẩn chúng trên validation/profiling, khóa trước main experiment. Tỷ lệ mẫu và histogram nhãn chỉ là proxy representation, không bảo đảm chất lượng học quên hoặc chất lượng IDS.

Điểm khác cần ghi trong báo cáo: thesis có biểu thức tổng quát tối ưu chất lượng Q dưới budget; đề cương mô tả chọn thời gian nhỏ trong ràng buộc đại diện. Starter thực hiện cách thứ hai. Không gọi nó là tối ưu Q hoặc tối ưu retrain distance. Nếu muốn tối ưu Q, phải xây predictor chất lượng và kiểm chứng riêng; không được dùng retrain checkpoint tương lai để chấm candidate online.

Ví dụ chạy:

```powershell
docker compose -f compose.dt.yaml run --rm central run --case /case/case.json --initial /case/after_local.npz --twin /runs/twin_seed42.json --output /runs/adaptive_seed42 --budget-seconds 60 --budget-bytes 10000000 --policy dt_adaptive --max-js 0.05 --min-coverage 0.2 --margin 1.25
```

`60` giây và `10 MB` chỉ minh họa cú pháp. Chọn tight/moderate/loose từ calibration và baseline thực đo, không bịa ngưỡng thời gian. `NO_FEASIBLE_PLAN` là kết quả hợp lệ; không tự nới budget hoặc giảm số vòng để “chạy cho xong”.

Ngân sách starter là **recovery + control overhead đo trong coordinator**, chưa bao gồm FU local, KD, provisioning, setup/calibration ban đầu hoặc evaluator offline. Nếu đề tài dùng ngân sách end-to-end UaaS:

```text
B_recovery = B_total - T_FU_measured_or_reserved - T_DT_control_reserved - T_other_reserved
```

Sau đó báo cả `T_total_actual` từ lúc nhận request đến checkpoint/report commit. Nếu chi phí bắt buộc đã vượt ngân sách, trả không khả thi trước recovery. Không báo budget recovery như budget toàn bộ FU.

## 9. Bước 6 — Cách round phân tán chạy và kiểm tra tính đúng

Một round trong `dt/coordinator.py`:

1. Tạo `base_hash` từ global weights hiện tại.
2. Gửi cùng weights đến các node trong S bằng thread pool, cùng model/data version được chỉ định.
3. Mỗi node clone model, gọi đúng `ClientWorker.train`, train R riêng, gửi state + count + telemetry.
4. Chờ barrier: đủ mọi participant mới tổng hợp.
5. Kiểm tra `base_hash`, job ID, data version, số mẫu/hist, tên tensor, shape/dtype và finite.
6. Gọi đúng `aggregate` upstream với retained counts; không lấy trung bình đều.
7. Ghi log, cập nhật EWMA, thực hiện vòng tiếp theo từ global model mới.
8. Sau đủ R vòng mới ghi `recovered.npz`.

Seed giữ công thức upstream: `seed + round_id*10000 + client_id*100`. Adam được khởi tạo lại mỗi client/round đúng core cũ. Tất cả policy phải bắt đầu từ cùng checkpoint sau FU.

RPC dùng NPZ thay vì `torch.load` trên dữ liệu mạng; không nhận pickle tùy ý. Worker có token, lock, disk cache theo job ID + request fingerprint: request trùng trả cùng kết quả, cùng ID nhưng khác nội dung trả 409. Cần snapshot/cache lifecycle nếu dùng dữ liệu thật.

Khi một node lỗi, starter abort round, không tự FedAvg tập còn lại. Không có transaction phục hồi coordinator sau crash; report FAILED/RUNNING không phải checkpoint model hoàn chỉnh. Sau HTTP timeout, worker có thể vẫn train; kiểm tra `/state` hết busy trước khi chạy lại. Starter không retry tự động hoặc cancel job đang chạy.

Do dự báo chỉ là ước lượng, execution vẫn có thể vượt budget. Starter hoàn thành đủ số vòng đã khóa và ghi `budget_violated`; chưa có bảo đảm deadline cứng. Nếu cần deadline cứng, thêm admission margin + deadline propagation + hủy tác vụ/cooperative cancellation và định nghĩa rõ cách xử lý checkpoint không hoàn tất.

## 10. Bước 7 — Những phép đo phải tách biệt

| Chỉ số | Nguồn | Giới hạn |
|---|---|---|
| Round wall time | Đồng hồ monotonic ở coordinator | Không cộng thời gian client song song |
| Tổng compute seconds | Cộng `compute_seconds` của client | Là resource cost, khác latency |
| HTTP body bytes | Len request + response | Không phải wire bytes |
| Tensor bytes | Tổng numel × element_size | Chưa tính metadata/NPZ/HTTP |
| Wire bytes | tcpdump/interface counter trong cửa sổ cô lập | Cần scope và không đếm hai đầu |
| Memory | sampled process peak RSS | Không phải peak container/cgroup; có thể bỏ peak ngắn |
| DT overhead | state probes, planner, predictor update, persistence | Phải bổ sung instrumentation chi tiết khi báo paper |
| UaaS total time | Timestamp nhận request → commit final report | Starter CLI chỉ có recovery boundary |

Trong report starter, `actual_body_bytes` chỉ gồm model train request/response bodies; chưa bao gồm `/state`, HTTP headers, TCP, TLS và retransmission. `budget_bytes` hiện cũng theo cùng scope body. Nếu khóa luận yêu cầu tổng truyền thông, mở rộng counter/transport trước khi báo kết quả end-to-end.

Đo wire bytes trên Linux có thể dùng tcpdump tại central trong một cửa sổ chỉ có một run. Nếu chưa cài: `sudo apt-get install tcpdump tshark`. Ví dụ central EC2, interface xác định bằng `ip -br addr`:

```bash
sudo tcpdump -i ens5 -nn -s 0 -w run.pcap 'tcp port 8000'
# Sau run: Ctrl+C ở phiên capture
capinfos run.pcap
tshark -r run.pcap -T fields -e frame.len | awk '{s+=$1} END {print s}'
```

`ens5` là ví dụ, phải thay theo interface thật. Tổng `frame.len` là bytes frame capture quan sát được, không gồm mọi overhead tầng vật lý; ghi rõ phương pháp. Không cộng cùng traffic đo ở central rồi cộng thêm toàn bộ client. Docker cùng bridge cần capture đúng interface, tránh `-i any` gây bản ghi trùng; chỉ đo trên physical NIC có thể bỏ toàn bộ traffic giữa container cùng host.

Trên Linux cgroup v2, lấy `memory.peak` nếu có và kiểm tra scope/reset trước run; shared host RSS không thay thế được memory container. Lưu cả Docker stats/CloudWatch host metrics làm bối cảnh, không dùng CPU% host như CPU% của riêng tác vụ.

Sai số predictor theo các run completed:

```text
MAE = mean(abs(actual_seconds - predicted_seconds))
WAPE = sum(abs(actual-predicted)) / sum(actual)
MAPE = mean(abs(actual-predicted) / actual), loại/định nghĩa riêng actual gần 0
ViolationRate = số completed run vượt ít nhất một budget / số completed run
AdmissionRate = số run được nhận / tổng request
CompletionRate = số completed / số admitted
```

Báo failure/no-feasible/timeout riêng để không làm đẹp violation rate bằng cách bỏ ca khó. Prediction phải lưu trước execution; không cập nhật dự báo bằng kết quả của chính run đó rồi gọi là predicted value. Starter lưu `twin_before.json`, `plan` và logs thô; feedback dùng cho lần sau. Các bản `twin_after.json` dùng cho audit.

## 11. Bước 8 — Chạy đối chứng và kịch bản lỗi trên Docker

Các policy đã có:

```text
all           tất cả client, kể cả khi dự báo vượt budget
 target       chỉ target, để kiểm tra giới hạn tối giản
random_k      target + k-1 node ngẫu nhiên theo seed
fastest_k     target + k-1 node nhanh nhất theo twin
largest_k     target + k-1 node nhiều retained samples nhất
 dt_static    chọn bằng twin nhưng không cập nhật twin sau execution
 dt_adaptive  chọn bằng twin và cập nhật sau execution
```

Khoảng trắng đầu một số dòng trên chỉ để trình bày, không thuộc tên policy. Baseline all/target/random/greedy được phép vượt budget để đo trade-off, không gọi chúng là feasible khi report ghi `predicted_feasible=false`. Prediction field của baseline không khả thi có thể thiếu trong starter; nếu cần tính prediction MAE cho mọi baseline, thêm hàm estimator trả dự báo cho cả candidate bị loại.

Ví dụ đối chứng cùng checkpoint:

```powershell
docker compose -f compose.dt.yaml run --rm central run --case /case/case.json --initial /case/after_local.npz --twin /runs/twin_seed42.json --output /runs/all_seed42 --policy all --budget-seconds 60 --budget-bytes 10000000
docker compose -f compose.dt.yaml run --rm central run --case /case/case.json --initial /case/after_local.npz --twin /runs/twin_seed42.json --output /runs/fastest_seed42 --policy fastest_k --k 3 --budget-seconds 60 --budget-bytes 10000000
```

So sánh `after_local` với `original` để có FU+recovery và retain-only+recovery. So sánh `after_kd` chỉ trong ablation KD. Không thay checkpoint hoặc số vòng âm thầm giữa các policy.

**Chống contamination của twin:** trước mỗi nhánh paired comparison, copy cùng snapshot calibration thành twin riêng; policy adaptive chỉ được cập nhật bản của nhánh đó. Thí nghiệm online drift riêng thì giữ chuỗi feedback theo thời gian, khóa workload order và so hai controller trên chuỗi điều kiện tương ứng. Không chạy adaptive trước rồi cho static dùng twin đã được adaptive học từ tương lai.

### Heterogeneous CPU

Ví dụ tạo `compose.hetero.yaml`:

```yaml
services:
  client0:
    cpus: "0.5"
  client1:
    cpus: "1.0"
  client2:
    cpus: "0.75"
  client3:
    cpus: "1.5"
  client4:
    cpus: "1.0"
```

```powershell
docker compose -f compose.dt.yaml -f compose.hetero.yaml up -d --wait client0 client1 client2 client3 client4
```

Recalibrate nếu đây là profile ổn định mới. Để kiểm tra adaptation khi drift, hiệu chuẩn trước, đổi profile tại thời điểm định trước rồi đo khả năng cập nhật; không dùng calibration sau drift cho controller tĩnh.

### Bandwidth/latency

Tạo override chỉ cấp `NET_ADMIN` cho node cần điều chỉnh, ví dụ `compose.netem.yaml`:

```yaml
services:
  client2:
    cap_add: [NET_ADMIN]
```

```powershell
docker compose -f compose.dt.yaml -f compose.netem.yaml up -d --wait client2
docker compose -f compose.dt.yaml -f compose.netem.yaml exec -u 0 client2 tc qdisc replace dev eth0 root netem delay 40ms 5ms rate 10mbit
# Sau kịch bản
docker compose -f compose.dt.yaml -f compose.netem.yaml exec -u 0 client2 tc qdisc del dev eth0 root
```

`eth0` phải kiểm tra bằng `ip -br addr` trong container. Ví dụ trên chỉ định hình **egress của client2**, chưa phải hai chiều. Muốn mô phỏng downlink riêng, định hình egress tương ứng tại sender/router hoặc dùng IFB ingress; không áp qdisc central trên mọi traffic rồi gọi đó là downlink riêng của client2. `delay 40ms` một chiều không tự có nghĩa RTT đúng 40ms. Docker Desktop/kernel có thể không hỗ trợ qdisc; nếu lệnh lỗi, ghi lại và chuyển phần impairment sang Linux host/EC2 hỗ trợ, không thay bằng sleep rồi gọi đó là bandwidth thực.

### Offline và stale

```powershell
docker compose -f compose.dt.yaml stop client2
# Chạy một run mới: starter hiện abort nếu bất kỳ worker trong pool offline ở bước validate_pool.
docker compose -f compose.dt.yaml up -d --wait client2
```

Đây là failure test, chưa phải hỗ trợ churn hoàn chỉnh. Để thích nghi offline theo BIPPO, cần registry availability và planner bỏ candidate chứa node offline; target offline phải trả không thể thực thi. Starter fail-fast toàn pool giúp không tạo kết quả âm thầm với ownership thay đổi.

Test stale bằng chỉnh `received_at` trong **bản twin thử nghiệm riêng** về quá khứ, hoặc chạy `--ttl` rất nhỏ. Không sửa clock OS. TTL đang dùng thời điểm coordinator nhận số đo; đồng bộ NTP/chrony khi phân tích timestamp liên máy.

## 12. Bước 9 — Thiết kế thí nghiệm đủ để viết khóa luận

Bắt đầu nhỏ: 1 seed, 1 request, profile ổn định, 2 recovery rounds. Sau khi artifact/checkpoint/transport đúng, mở rộng:

1. Ít nhất 3 seed, tốt hơn nếu đủ kinh phí; báo CI theo seed/request chứ không coi mỗi row là một replicate độc lập.
2. Hai mức non-IID từ protocol hiện có, nhiều target, các fraction quy định trước; không chọn target/fraction sau khi xem test.
3. Budget tight/moderate/loose từ baseline đo thật; bao gồm trường hợp không feasible.
4. Stable, CPU heterogeneous, bandwidth impairment, workload drift, stale telemetry, node failure.
5. `all`, `target`, `random_k`, `fastest_k`, `largest_k`, `dt_static`, `dt_adaptive`; oracle chỉ dùng offline để đánh giá upper bound.
6. Giữ số vòng/local epochs/batch và cơ chế FU cố định để phân tích riêng client selection.

Giữ cả no-op, original, fresh retrain, retain-only và FU variants trong evaluator. `recovered.npz` có thể chuyển về model bằng adapter sau, chạy **trên máy evaluator offline**, không chạy trên central:

```python
from pathlib import Path
from dt.codec import unpack
from src.fl.fedavg_core import model, predict
weights, metadata = unpack(Path('runs/dt/adaptive_seed42/recovered.npz').read_bytes())
m = model(input_dim, [64, 32], 42)  # input_dim lấy từ case đã kiểm tra
m.load_state_dict(weights)
probs = predict(m, X_evaluation)  # chỉ evaluator được đọc dữ liệu đánh giá
```

Tích hợp vào `src/eval/metrics.py` hiện có, sử dụng đúng evaluation IDs và attack calibration của paired pilot. Không train lại MIA/tune threshold sau khi nhìn test. Các số đo cần có:

- Utility: macro-F1, precision/recall, PR-AUC/AP, FPR toàn cục và per-client.
- FU: prediction MAE/JS/KL/disagreement tới retrain trên F/R/test, MIA + positive control, so với retain-only.
- DT: time MAE/WAPE, violation/admission/completion rate, time/body/wire/compute/memory/DT overhead.
- Robustness: failure cases, state staleness, drift recovery, calibration/setup cost.

Không tuyên bố “FU tốt” chỉ vì macro-F1 giữ nguyên hoặc thời gian giảm. Pilot hiện có MIA control yếu; kết luận forgetting có thể vẫn inconclusive. Có thể báo cáo độc lập “DT cải thiện dự báo/điều phối chi phí trên workload FU cố định” nếu số đo hỗ trợ.

Oracle không được chạy mọi candidate thật bên trong một request online rồi chỉ tính chi phí của candidate tốt nhất; nếu làm offline thì chi phí probing phải ghi riêng.

## 13. Bước 10 — Chuyển đúng kiến trúc sang EC2

**Không dùng Docker Compose một host để giả lập đa host trên AWS.** Mỗi client EC2 chạy một worker container, central EC2 chạy coordinator+DT; chúng giao tiếp bằng private IP trong VPC. Giữ cùng image digest, shard, checkpoint, seed và thuật toán; thay URL và calibration.

### 13.1 Chuẩn bị tài khoản và mạng

Ví dụ region `ap-southeast-1`; có thể thay theo tài khoản/quota/chi phí. Không có số giá cố định trong hướng dẫn; xem pricing thực tế trước khi launch.

1. AWS Console → VPC → tạo VPC, ví dụ `10.42.0.0/16`, bật DNS resolution/hostnames.
2. Tạo subnet cùng AZ, ví dụ `10.42.1.0/24`; cùng AZ trước để giảm biến số mạng. Sau đó mới khảo sát nhiều AZ.
3. Chọn một đường outbound rõ ràng:
   - Lab đơn giản: subnet public có route tới Internet Gateway, EC2 có public IPv4 để tải image/SSM. Mọi RPC dùng private IP; inbound khóa như bên dưới.
   - Private subnet: dùng NAT hoặc VPC endpoints đủ cho SSM/ECR/S3. NAT/endpoints có chi phí riêng; private subnet không tự tải image được nếu thiếu đường đi.
4. Tạo hai security group `dt-central-sg` và `dt-worker-sg`.
5. Worker SG: inbound TCP 8000 **source = central SG**, không phải `0.0.0.0/0`.
6. Central không cần inbound model-RPC. Quản trị bằng SSM Session Manager; nếu chọn SSH thì chỉ cho TCP 22 từ IP quản trị `/32` và dùng key.
7. Cả hai cần outbound hợp lệ cho RPC private, DNS, registry và AWS APIs. Endpoint SG nếu có phải cho HTTPS từ instance SG.
8. Tạo IAM instance profile có `AmazonSSMManagedInstanceCore`; ECR pull read-only, CloudWatch nếu dùng. Provision dữ liệu qua S3 thì cấp worker quyền đọc **prefix của chính nó**, central chỉ prefix metadata/checkpoint của central.
9. EC2 launch chọn AMI Ubuntu Server 24.04 LTS, x86_64 cho image amd64; kiểm tra AMI owner chính thức trong Console. Không copy AMI ID từ region khác.
10. Tạo 6 instance: `dt-central`, `dt-client-0` ... `dt-client-4`. Bắt đầu cùng loại instance, ví dụ profile 2 vCPU/4 GiB để thử MLP nhỏ, rồi đo để chỉnh. Đây là cấu hình thử, không bảo đảm đủ cho mọi dataset.
11. Gắn đúng SG/instance profile, bật IMDSv2 required, EBS encryption, tag `Project=budget-aware-dt`, `RunGroup=...`.
12. Ghi private IPv4 từng node. Đảm bảo SSM agent online hoặc SSH hoạt động trước khi provisioning.

Nếu chọn T-family burstable, CPU credit có thể làm thời gian drift theo phiên chạy hoặc sinh chi phí khi unlimited; lưu `CPUCreditBalance`, `CPUSurplusCreditBalance` và policy. Dùng instance non-burstable hoặc kiểm soát credit nếu mục tiêu là đo hiệu năng ổn định. So sánh các cấu hình instance trong thí nghiệm riêng, không trộn với đánh giá thay thuật toán.

SG là ranh giới mạng; token HTTP trong starter không mã hóa payload. Với dữ liệu/mô hình thật hoặc triển khai ngoài lab được kiểm soát, đặt TLS/mTLS qua reverse proxy hoặc HTTPS endpoint nội bộ. Private IP không tự cung cấp TLS. Không công khai `/train` lên Internet.

### 13.2 Cài Docker trên Ubuntu

Các lệnh dưới đây là **Bash trên từng EC2**, không phải PowerShell. Dùng repository Docker chính thức:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $VERSION_CODENAME stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker version
sudo docker compose version
```

Cài AWS CLI v2 theo hướng dẫn chính thức nếu image chưa có. Xác minh `aws sts get-caller-identity`; dùng instance role, tránh đặt access key dài hạn vào image/.env. Các lệnh phía dưới dùng `sudo docker`; không cần sửa quyền nhóm Docker để làm lab.

### 13.3 Đưa image vào ECR

Chạy tại máy build có Docker và AWS CLI được cấu hình đúng quyền. Ví dụ Bash:

```bash
export AWS_REGION=ap-southeast-1
export AWS_ACCOUNT_ID=REPLACE_ACCOUNT_ID
export DT_IMAGE="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/budget-dt:v1"
aws ecr create-repository --repository-name budget-dt --region "$AWS_REGION" --image-tag-mutability IMMUTABLE
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
docker build --platform linux/amd64 -f Dockerfile.dt -t "$DT_IMAGE" .
docker push "$DT_IMAGE"
aws ecr describe-images --repository-name budget-dt --image-ids imageTag=v1 --region "$AWS_REGION"
```

Nếu repository đã có thì bỏ bước create; tag immutable đã tồn tại thì dùng version mới. Ghi lại `imageDigest` và dùng `.../budget-dt@sha256:...` trên mọi EC2. Không dùng `latest` cho kết quả khoa học. Build context đã có `.dockerignore` và Dockerfile chỉ copy source cần thiết; vẫn kiểm tra image không chứa raw data hoặc `.env`.

Trên mỗi EC2, lấy thông tin role và login registry:

```bash
export AWS_REGION=ap-southeast-1
export AWS_ACCOUNT_ID=REPLACE_ACCOUNT_ID
aws sts get-caller-identity
aws ecr get-login-password --region "$AWS_REGION" | sudo docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
```

### 13.4 Cấp shard cho từng worker

Ở máy offline, có thể upload từng `provision/clientN/` vào S3 prefix riêng trong bucket private, SSE được cấu hình, block public access. Không upload raw/global train lên central.

Ví dụ trên **client0 EC2**, sau khi prefix đã được upload và role được cấp đúng quyền:

```bash
sudo mkdir -p /opt/budget-dt/data
sudo chown ubuntu:ubuntu /opt/budget-dt /opt/budget-dt/data
aws s3 sync s3://REPLACE_BUCKET/dt-case-a/client0/ /opt/budget-dt/data/
```

Nếu OS user khác `ubuntu`, thay đúng user. Chép `compose.ec2-worker.yaml` vào `/opt/budget-dt/`. Tạo `.env` bằng editor với các giá trị thật:

```dotenv
DT_IMAGE=REPLACE_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/budget-dt@sha256:REPLACE_DIGEST
CLIENT_ID=0
DT_TOKEN=REPLACE_RANDOM_TOKEN_SHARED_WITH_CENTRAL
BIND_IP=REPLACE_CLIENT0_PRIVATE_IPV4
LOCAL_DATA=/opt/budget-dt/data
```

```bash
cd /opt/budget-dt
chmod 600 .env
sudo docker compose --env-file .env -f compose.ec2-worker.yaml config --quiet
sudo docker compose --env-file .env -f compose.ec2-worker.yaml up -d --wait
sudo docker compose --env-file .env -f compose.ec2-worker.yaml logs --tail 30
```

Lặp lại client1...client4 với đúng shard, ID và private IP. Không dùng cùng shard cho mọi node. Bind `BIND_IP` lên địa chỉ private; security group vẫn bắt buộc.

### 13.5 Khởi chạy central EC2

Chép **chỉ** `provision/central` thành `/opt/budget-dt/central`, tạo `/opt/budget-dt/runs`, và file `.env` chứa `DT_TOKEN`. Export biến `DT_IMAGE` bằng digest như worker.

Sửa `central/case.json`: thay `http://clientN:8000` bằng `http://PRIVATE_IPV4_N:8000`. Có thể dùng Python script trên central:

```python
import json
from pathlib import Path
p = Path('/opt/budget-dt/central/case.json')
case = json.loads(p.read_text())
private_ips = ['REPLACE_IP0','REPLACE_IP1','REPLACE_IP2','REPLACE_IP3','REPLACE_IP4']
for c in case['clients']:
    c['url'] = f"http://{private_ips[c['id']]}:8000"
p.write_text(json.dumps(case, indent=2))
```

Các IP phải được thay trước khi chạy. `data_version` giữ nguyên nếu byte shard không đổi. Tạo twin **mới cho EC2**, không dùng lại twin Docker:

```bash
sudo docker run --rm --env-file /opt/budget-dt/.env \
  -v /opt/budget-dt/central:/case:ro -v /opt/budget-dt/runs:/runs \
  --entrypoint python "$DT_IMAGE" -m dt.coordinator calibrate \
  --case /case/case.json --initial /case/after_local.npz \
  --twin /runs/twin_ec2.json --output /runs/calibration_ec2

sudo docker run --rm --env-file /opt/budget-dt/.env \
  -v /opt/budget-dt/central:/case:ro -v /opt/budget-dt/runs:/runs \
  --entrypoint python "$DT_IMAGE" -m dt.coordinator run \
  --case /case/case.json --initial /case/after_local.npz \
  --twin /runs/twin_ec2.json --output /runs/adaptive_ec2 \
  --policy dt_adaptive --budget-seconds 60 --budget-bytes 10000000
```

Nếu dùng private DNS nội bộ thay IP, cần cấu hình DNS thực và đảm bảo identity mapping; Docker service name `client0` không tự có ý nghĩa ở EC2 khác.

### 13.6 Giám sát, lưu kết quả và dọn tài nguyên

- CloudWatch: CPU/network/credit của EC2; thêm CloudWatch Agent nếu cần memory/disk guest metrics. Process/container metrics lấy thêm từ worker/cgroup.
- S3 results: chỉ upload report/checkpoint được cho phép; tách prefix theo commit/image/seed/request/profile/policy. Không gửi request row IDs hoặc raw data vào log công khai.
- Lưu instance type, vCPU quota, RAM, AMI, OS/kernel, AZ/region, image digest, seed, data/model hash, config, twin snapshot và timing boundary.
- Tạo AWS Budget notification cho tài khoản/project và xem Billing/Cost Explorer. AWS Budgets không phải nút chặn chi phí tức thì; dữ liệu cập nhật trễ. Không dùng nó thay admission control của request.
- Kết thúc thí nghiệm: stop/terminate đúng instance của project, kiểm tra EBS/snapshot, public IPv4, NAT, endpoints, ECR/S3/CloudWatch retention. Stop EC2 không đồng nghĩa mọi khoản phí dừng. Không xóa snapshot cần thiết cho tái lập trước khi archive kết quả.

Không có tài nguyên AWS nào được tạo bởi bộ hướng dẫn này. Các lệnh cloud cần chạy bằng tài khoản/quyền của bạn sau khi điền các placeholder.

## 14. Khi nào mới thêm PPO theo BIPPO?

Không bắt buộc dùng PPO để gọi hệ thống là Digital Twin. Với 16 subset, exhaustive planner là baseline dễ giải thích và kiểm chứng. Nếu hội đồng yêu cầu BIPPO-inspired selection hoặc quy mô tăng, thêm policy sau khi DT/cost/transport đã đáng tin:

1. Xây môi trường RL từ trace profiling/training; tách seed/profile để train/validate/test.
2. Local observation gợi ý: retained count/hist proxy, validation utility/loss, expected compute/bytes, availability, tham gia vòng trước, state age; normalize bằng statistics từ training episodes.
3. Actor từng client trả participation score. Sampler kiểm tra **cả tập S** theo predictor round latency/bytes, không chỉ cộng cost_i như năng lượng BIPPO.
4. Mask offline/stale, bắt buộc target; thiếu ngân sách tối thiểu trả no feasible.
5. Reward chỉ dựa validation/recovery proxy có sẵn online. Ví dụ `delta_validation_macro_f1 - lambda*budget_excess - eta*control_cost`; đây là adaptation cần ablation, không phải reward nguyên bản paper.
6. Không dùng distance tới retrain thật trong reward vận hành nếu không có retrain. Có thể dùng nó offline để đánh giá, tách biệt rõ.
7. Ghi cả training/inference cost của RL, số probe và cost xây simulator; không tính mỗi inference rồi kết luận tiết kiệm tổng thể.
8. Benchmark với exhaustive/greedy/random trên cùng workload/budget; không giả định PPO tốt hơn.
9. Với on-policy PPO, phải nhất quán action distribution/log-prob với sampler và feasibility masking. Không thay action tùy ý sau sample rồi dùng log-prob của action cũ. Mở rộng này cần thiết kế thuật toán riêng, không phải chỉ thêm một mạng actor.

Bộ starter hiện **không triển khai PPO**, cũng không tuyên bố tái hiện BIPPO. Phần chi tiết PPO của paper không phải prerequisite cho milestone DT của đề cương.

## 15. Hoàn thiện UaaS sau khi DT/recovery ổn định

Luồng đích cho yêu cầu xóa mới:

```text
RECEIVED → VALIDATED → LOCAL_FU_RUNNING → RETAINED_DATA_COMMITTED
         → TWIN_REFRESHED → PLANNED → RECOVERY_RUNNING
         → OPERATIONALLY_VERIFIED → COMPLETED
```

Các trạng thái kết thúc khác: `REJECTED`, `NO_FEASIBLE_PLAN`, `FAILED`, `NEEDS_REVIEW`. Không đánh dấu scientific forgetting success chỉ vì operational verification passed.

Cách mở rộng code, theo thứ tự:

1. Tạo `src/service/api.py`: `POST /unlearning-requests`, `GET /requests/{id}`; trả 202/job ID. Không train dài trong request HTTP nhận job.
2. SQLite/PostgreSQL lưu request model/data version, target, idempotency key, status transitions; một coordinator lease bảo đảm không có hai recovery cùng ghi global model.
3. Target nhận opaque request handle qua kênh riêng; F IDs được resolve cục bộ. DT chỉ thấy request hash/count và metadata cần thiết. Không gửi raw F lên server.
4. Worker target load original checkpoint và gọi nguyên hàm có sẵn:

```python
from src.fu.conpfu_inspired import alternating, distill
# original: model đúng phiên bản; fw: local worker cho F; rw: local worker cho R
corrected, fu_log = alternating(original, fw, rw, fu_config, seed)
if use_kd:
    corrected, kd_log = distill(original, corrected, rw, fu_config, seed)
# Chỉ gửi corrected weights + aggregate telemetry, không gửi fw/rw
```

5. Commit corrected checkpoint và phiên bản retained data theo một transaction/journal rõ ràng. Retained membership phải loại F trước mọi recovery. Bản triển khai chỉ đọc retained; restart không được load lại shard cũ.
6. Sau commit, drop operational references/tạm của F, xử lý file tạm/cache/snapshot/backups theo chính sách đã định. `del fw` hoặc xóa file đơn lẻ không tự bảo đảm dữ liệu đã được xóa khỏi mọi lớp storage; không gọi đó là certified deletion. Evaluator nghiên cứu có vùng quyền riêng để đo trên F, tách khỏi dịch vụ.
7. Đo/ước lượng chi phí FU local, cập nhật budget còn lại; refresh twin rồi lập recovery plan như bộ starter.
8. Kết quả checkpoint mới chỉ được publish bằng compare-and-swap từ `base_model_hash` đã nhận; cập nhật registry/audit và thông báo trạng thái job.
9. Sau crash, đọc journal để resume/reconcile đúng stage; không chạy FU hoặc xóa dữ liệu lại một cách mù quáng. Add retry theo idempotent stage, heartbeat, cooperative cancellation và quota.
10. Tách evaluator offline khỏi request path nếu nó dùng retrain/F/test; theo dõi operational metrics online và scientific verification trong report nghiên cứu.

Có thể triển khai service API/DB trong central cùng máy trước, rồi tách khi có nhu cầu. Không cần Kubernetes/EKS để hoàn thành prototype này.

## 16. Tiêu chí hoàn tất từng milestone

| Mốc | Điều kiện thực sự hoàn tất |
|---|---|
| M1: distributed FL adapter | Cùng weights/counts/seeds cho kết quả đúng; raw ở client; barrier/version/idempotency/lỗi được kiểm tra |
| M2: state twin | Số đo thật có timestamp/version; stale được phát hiện; snapshot tái lập |
| M3: prediction | Calibration/past-only fitting; lưu dự báo trước chạy; MAE/WAPE + drift/failure được báo |
| M4: planner | 16 subsets; target/budget/representation; no-feasible; action thật điều khiển worker |
| M5: Docker study | Baseline paired, nhiều seed/profile; overhead/body/wire/memory phân biệt; không bỏ ca xấu |
| M6: EC2 study | Cùng image/core, private networking, calibration mới, artifact/chi phí cloud được ghi |
| M7: UaaS | Job lifecycle, retained version transaction, retries/recovery, request boundary và deletion policy rõ |

### Trạng thái kiểm chứng của bundle

Xem `VALIDATION.md` đi kèm để biết chính xác tests và môi trường đã chạy. Không có full ToN-IoT dataset/pilot checkpoint thực tế trong checkout công khai dùng để kiểm thử bộ này; chưa chạy full FU hoặc deploy AWS. Không có Docker executable trên máy thực hiện, nên image build/Compose/`tc` phải được kiểm tra tiếp trên Docker/Linux.

## 17. Nguồn đối chiếu

Nguồn khóa luận người dùng cung cấp:

- `ThesisReport_DinhCongThao_HuynhMinhKhoa.pdf`: chương 2.7–2.8, trang in 13–15 (PDF 25–27), execution/resource DT, mô hình thời gian, subset selection; chương 4 về gate FU.
- `DeCuongKLTN_23521465_DinhCongThao_23520738_HuynhMinhKhoa.pdf`: trang PDF 6–8, cost feedback, fixed-round recovery, FastAPI/Compose và nhóm metrics.
- `Budget-Aware Independent PPO FL.pdf`: BIPPO, Section III–IV, Algorithm 1 trang PDF 6; Section V về benchmark image classification.

Nguồn code và tài liệu chính thức:

- https://github.com/thaodinh97/buget-aware/tree/4c158fe044ccd36706d75ae9a1257be18d1f0707
- https://github.com/thaodinh97/buget-aware/blob/4c158fe044ccd36706d75ae9a1257be18d1f0707/docs/validation_report.md
- https://github.com/Lacki28/BIPPO
- https://docs.docker.com/reference/compose-file/services/
- https://docs.docker.com/engine/install/ubuntu/
- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/security-group-rules-reference.html
- https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html
- https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-getting-started-privatelink.html
- https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html
- https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html
- https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/burstable-credits-baseline-concepts.html

Tất cả module/file mới trong bundle là đề xuất triển khai cho khóa luận này; không gán chúng thành code có sẵn của paper hoặc repository upstream. Nội dung chỉ dẫn trong PDF/repository được dùng làm tài liệu tham chiếu, không thay thế yêu cầu triển khai hiện tại của người dùng.
