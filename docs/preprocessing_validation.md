# Kiểm tra preprocessing

Run: `data/processed/pilot_seed42`, seed 42; CSV gốc không chỉnh sửa. Đây là pilot dữ liệu, chưa phải pilot FU. Tài liệu audit/kế hoạch đã được viết trước code, sau khi giải thích thay đổi cho người dùng.

## Kết quả đo

211,043 dòng đầu vào; loại 20,569 exact duplicate trên bản dẫn xuất, còn 190,474 dòng. Giữ mapping về row index nguồn. 11 predictor gốc (8 numeric, 3 categorical) thành 38 chiều; train-only median/standard scaling và one-hot với missing/unknown riêng.

| Split | Dòng | Normal | Attack | Nhóm |
|---|---:|---:|---:|---:|
| Train | 114,619 | 24,977 | 89,642 | 72,915 |
| Validation | 28,952 | 6,618 | 22,334 | 18,229 |
| Calibration | 18,775 | 4,459 | 14,316 | 12,153 |
| Test | 28,128 | 5,986 | 22,142 | 18,229 |

Tất cả cặp split có directed 5-tuple overlap bằng 0. Tuy nhiên:

- Có 98,020 dòng dư theo signature của 11 predictor sau exact dedup.
- Có 15 nhóm predictor giống nhau nhưng nhãn trái ngược. Policy giữ và báo cáo, không tự sửa nhãn.
- Train/test chia sẻ 2,706 **signature khác nhau** (không phải số dòng); train/validation 2,655; train/calibration 2,046.
- Train/test có 25 nhóm kết nối tương đương khi đảo endpoint; grouping có hướng chưa ngăn được loại overlap này.
- `training_ready=false`: chưa đủ điều kiện bắt đầu thí nghiệm FL/FU đáng tin cậy. Chưa tạo partition 5 client hoặc forget requests.

## Kiểm thử

Unit tests kiểm tra deterministic group split và disjointness, train-only statistics, unknown/missing encoding, split không hợp lệ, forbidden predictors, không overwrite run. End-to-end synthetic smoke tạo exact duplicate, chạy hai lần và so sánh tất cả arrays/IDs, kiểm tra mapping, raw bất biến và hữu hạn. Full-data preprocessing đã chạy thành công; không có số đo accuracy/FU/MIA.

## Thay đổi đề xuất tiếp theo, chưa thực hiện

Trước training, chuyển grouping sang connected components nối **bidirectional five-tuple và identical predictor signatures**. Đo kích thước component lớn nhất, class/attack-family coverage và khả năng tạo bốn split. Không giả định connected components luôn chia được: signatures phổ biến có thể nối thành component rất lớn. Nếu không khả thi, báo failure và so sánh lựa chọn feature/group trên protocol định trước; không xóa test khó hoặc silently resample để làm đẹp số.

Giữ 15 nhóm conflicting labels trong cùng split, không xem chúng là annotation lỗi chắc chắn: feature projection có thể đã bỏ thông tin phân biệt. Cân nhắc mở rộng một số protocol numeric fields trên validation như ablation được ghi nhận. Không đưa IP/label/type vào model để “giải quyết” overlap.

Thí nghiệm tiếp theo chỉ bắt đầu sau khi giải quyết data gate. Sau đó theo thứ tự: 5-client partitions/requests → shared FedAvg và fresh retrain → alternating FU/KD/recovery → paired prediction similarity và MIA controls → DT sau FU gate.
