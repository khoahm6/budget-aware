# Kiểm toán triển khai — 2026-10-18

> Audit ban đầu. Cập nhật: đã có preprocessing v3, five-client FedAvg/FU/retrain/evaluation và pilot thật; FU gate chưa đạt. Xem `validation_report.md`. Python hệ thống `D:/python/python.exe` có torch/sklearn, khác bundled runtime đã kiểm tra ban đầu.

## Phạm vi và hiện trạng

Đã kiểm tra đệ quy, kể cả file ẩn, tại `E:/budget-aware-code`. Workspace chỉ có hướng dẫn, ba PDF và `dataset/train_test_network.csv`; không có source Python, notebook, `fedavg_core.py`, preprocessing, checkpoint, manifest chạy, test, dependency file hay `.git`. Người dùng xác nhận không có repository mã nguồn khác. Vì vậy **không có thành phần code đã kiểm chứng để tái sử dụng**. Dữ liệu và tài liệu là đầu vào có thể dùng lại; thiết kế mới sẽ dùng chung một pipeline và một training/FedAvg core cho mọi baseline, không nhân bản logic.

Python bundled có pandas/numpy/pypdf; kiểm tra môi trường này không tìm thấy torch/sklearn. Điều này không chứng minh mọi Python trên máy đều thiếu chúng. Chưa cài dependency ML, chưa huấn luyện hoặc chạy FU.

## Dữ liệu đã đo trực tiếp

| Thuộc tính | Giá trị |
|---|---|
| File | `dataset/train_test_network.csv` |
| Kích thước | 29,902,775 bytes |
| SHA-256 | `26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974` |
| Số dòng / cột | 211,043 / 44 |
| Nhãn 0 / 1 | 50,000 / 161,043 |
| Bản ghi trùng toàn bộ, ngoài lần đầu | 20,569 |
| Nhóm 5-tuple có hướng | 121,526 |
| Nhóm nhiều hơn một dòng / lớn nhất | 21,983 / 3,363 dòng |
| Null theo parser pandas | 0 ở mọi cột |
| Xung đột nhãn trên toàn bộ 42 cột không phải label/type | 0 nhóm |

`type`: normal 50,000; tám họ backdoor/ddos/dos/injection/password/ransomware/scanning/xss mỗi họ 20,000; mitm 1,043. Chỉ `normal` có label=0. Không đưa `type` vào X.

Không có timestamp hoặc session ID. Có src/dst IP, src/dst port và proto để tạo nhóm kết nối đại diện; không gọi đó là phiên thực. Không thể tái tạo chronological split từ file này. Tên file không chứng minh đó là split chuẩn của tác giả; nguồn tải/version chưa được cung cấp, checksum chỉ xác định chính file đang dùng.

Không có null không có nghĩa dữ liệu đầy đủ: dấu `-` xuất hiện 132,032 lần ở service; 176,198 ở dns_query; 210,642 ở nhiều trường SSL. Cần phân biệt “không áp dụng” và missing; không biến mọi `-` thành số 0.

## Đối chiếu tài liệu gốc

- Đề cương PDF thực tế có hậu tố `(1).pdf`, khác tên `(1)(2).pdf` trong guide. Đã đọc 11 trang; phạm vi pp. 3–9 phù hợp FU mức mẫu, client vẫn online, MLP/FedAvg, 5 client, DT và container proof of concept. Đề cương p.6 cho phép fit transform trên calibration; guide ưu tiên train-only. Kế hoạch chọn train-only và ghi rõ khác biệt.
- ConPFU, pp. 6–8, Algorithm 1: loại update vòng cuối, alternating GA/GD có regularization và clipping, KD tại client với teacher là mô hình sau alternating và student là mô hình khởi đầu; KD dùng dữ liệu local D_i. P.6 cũng thảo luận dùng trực tiếp global model khi thiếu lịch sử và recovery như phần mở rộng. Guide vẫn yêu cầu gắn nhãn các lựa chọn này là adaptation của thí nghiệm. P.11 dùng CNN cho ToN-IoT, 20 client, Dirichlet 0.5, quên 50% một client. **Không có đặc tả preprocessing CSV đủ để tái lập y hệt**.
- FedKD-Unlearn, pp. 7–8, Algorithms 1–2: ascent có projection; teacher global gốc, student forget model; KD server dùng auxiliary labeled data. P.8 dùng MLP 128–64–32 cho ToN-IoT và train/validation 80/20 trong client với central test. P.10 đánh giá MIA và khoảng cách tham số tới retrain. Không có danh sách feature/encoder/scaler đầy đủ. Algorithm 1 không thể thay thế một triển khai DP có accounting; không suy ra bảo đảm DP cho đề tài từ paper.
- Không sao chép số đo kết quả của paper làm kết quả của repository. ConPFU dùng CNN, FedKD dùng ba hidden layer; MLP hai hidden layer là lựa chọn của đề cương.

## Thiếu chức năng

Chưa có split/transform có provenance, train-only client partitions, forget requests, MLP, sample-weighted FedAvg, fresh retraining, no-op, retain-only, alternating FU, KD, recovery, evaluation/MIA, immutable manifests, CLI/tests, telemetry hay DT. Sau khi triển khai preprocessing, cập nhật trạng thái từ báo cáo thực chạy; các phần còn lại vẫn là kế hoạch.

## Nguy cơ thiết kế cần giải quyết trước kết luận FU

1. Random row split làm rò rỉ duplicate/kết nối. Group-disjoint không đủ: bỏ IP/port có thể tạo feature signatures giống nhau giữa các split; phải đo và công bố.
2. Dedup toàn bộ dòng không loại hết các predictor giống nhau. Xung đột nhãn phải audit lại sau chọn feature; giữ và báo cáo, không tự sửa nhãn hoặc lọc để cải thiện metric.
3. Dấu `-`, kiểu số dạng chuỗi và outlier cần policy có version. Fit encoder/scaler trên toàn bộ CSV gây leakage.
4. Transform fit trên original training có thể giữ thống kê của F. Retraining với transform cố định là tham chiếu **có điều kiện trên biểu diễn cố định**, không phải xóa toàn bộ ảnh hưởng end-to-end. Báo cáo giới hạn; cân nhắc calibration bất biến ngoài phạm vi deletion cho nghiên cứu bổ sung.
5. Dirichlet alpha=0.1 dễ tạo client rỗng/đơn lớp. Reject rõ ràng, không âm thầm chuyển seed/partition. Chọn target theo metadata, không theo kết quả FU.
6. Quên một sample khi predictor-equivalent samples còn ở R có thể cho original/retrain rất gần nhau. Đây là tín hiệu không định danh, không phải FU thành công.
7. Retrain phải khởi tạo mới trên D\\F, giữ partition còn lại và cùng protocol, không warm-start hoặc repartition sau deletion. FedAvg weights phải cập nhật theo retained counts.
8. KD toàn D_i trong ConPFU dùng lại F; retain-only KD là adaptation cần ablation. Original-teacher KD của FedKD có nguy cơ đưa lại hành vi gốc. Đo cả trước và sau KD/recovery.
9. Không dùng forget accuracy hoặc MIA gần 50% làm chứng cứ duy nhất. MIA phải có controls và so với retrain; test không dùng chọn hyperparameter/planner.
10. Không suy ra thời gian mạng thật từ payload bytes hoặc sequential simulation. Ghi rõ phạm vi phép đo; chỉ gọi DT khi có trạng thái timestamp và phản hồi chi phí thật.

## Tái lập audit

Thống kê trên được tính trực tiếp bằng pandas trên toàn CSV; duplicate dùng toàn bộ cột; group dùng `(src_ip,src_port,dst_ip,dst_port,proto)`; conflict ban đầu dùng mọi cột trừ label/type. Các thuật toán và số liệu sau preprocessing được ghi riêng trong `data/processed/pilot_seed42/audit.json`. Không có checkpoint cũ để regenerate; lệnh training sẽ chỉ được cung cấp khi module training thực sự tồn tại.
