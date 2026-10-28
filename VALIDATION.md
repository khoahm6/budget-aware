# Validation — 27/10/2026

Upstream snapshot: `4c158fe044ccd36706d75ae9a1257be18d1f0707`.

## Đã chạy

- 10 tests nguyên bản của repository: PASS (4.027 giây).
- 5 tests bổ sung trong `tests/test_dt.py`: PASS (51.685 giây).
- Parse AST của mọi module Python mới: PASS.
- Parse YAML hai file Compose: PASS. Đây không phải `docker compose config` hoặc xác nhận runtime.

Năm test bổ sung:

1. HTTP round năm worker khớp tensor tuyệt đối với weighted FedAvg core hiện có trên synthetic data, cùng seed/config/counts.
2. Request trùng job ID trả lại cùng response; ID collision trả 409; thiếu token trả 401.
3. Planner luôn có target, trả không feasible khi thiếu bytes hoặc target telemetry stale.
4. CLI calibration → all-client recovery → adaptive recovery → no-feasible; ghi report/checkpoint đúng nhánh.
5. Export từ synthetic full-pilot artifacts: loại F khỏi mọi shard, đúng retained counts, central metadata không chứa row IDs.

HTTP integration khởi chạy năm uvicorn process trên localhost; không phải năm máy vật lý. Kiểm thử dùng PyTorch CPU thật và gọi trực tiếp core upstream, không mock train/FedAvg.

Môi trường: Windows, Python 3.13, PyTorch 2.6.0+cpu, FastAPI 0.162.0, Uvicorn 0.54.0, HTTPX 0.28.1, psutil 7.2.2. Dependency mới được cài vào thư mục scratch, không sửa môi trường Python gốc của người dùng.

## Chưa chạy

- Docker image build, Docker Compose runtime và `tc netem` vì máy không có Docker executable.
- Image Linux/Python 3.11 regression; cần kiểm tra lại sau build.
- Full ToN-IoT pipeline/export/recovery với dữ liệu thực, vì checkout công khai không chứa bộ dữ liệu/checkpoint đầy đủ.
- AWS provisioning, ECR push, IAM/SG/SSM, EC2 multi-host.
- Chưa đo prediction MAE/violation/speedup trên môi trường thí nghiệm thật.
- Chưa triển khai PPO, quality predictor, UaaS API/job queue, giao dịch deletion hoặc deadline cứng.

Không có kết quả khoa học hoặc tài nguyên AWS nào được tạo/khẳng định từ các kiểm thử này. `infrastructure_only` là nhãn chính xác cho starter; giữ gate FU hiện có.
