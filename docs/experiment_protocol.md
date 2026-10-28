# Protocol pilot FU — khóa trước khi chạy

Ngày 2026-10-18. Cấu hình `configs/pilot.json` được lưu vào manifest trước training. Pilot seed 42, alpha=0.5, 5 client; target là client lớn nhất theo số mẫu, không chọn bằng kết quả FU. Forget 10% và 50%, stratified theo nhãn và nested, floor mỗi lớp. Không đổi seed nếu partition lỗi.

## Dữ liệu và ranh giới

Group mới là thành phần liên thông của bidirectional five-tuple và identical feature signatures. Giữ 15 nhóm conflicting labels, nhưng toàn bộ nhóm ở cùng split; đây có thể là mất thông tin do projection feature, không tự kết luận nhãn sai. Split theo kích thước nhóm, largest-first và seeded tie order; không dùng nhãn để tối ưu split.

Kiểm tra bổ sung trước pilot phát hiện collisions sau float32. Phiên bản v3 gộp thêm nhóm có encoded vector giống nhau, rồi split/refit train-only cho đến khi không còn overlap (tối đa 10 lần, không hội tụ thì fail). Run này cần hai lần refinement. Việc này dùng covariates toàn bộ dataset để thiết kế nhóm, không dùng test labels/metrics để tune model; vẫn cần công bố đây là holdout có grouping dựa trên toàn dữ liệu, không chronological/external validation. Các run v1/v2 và smoke cũ chỉ là diagnostic, không dùng làm evidence FU.

Row-level Dirichlet chỉ trên train, nên cùng training group có thể ở nhiều logical clients. Đây là mô phỏng ownership; không tuyên bố client là những tổ chức vật lý độc lập. Source raw bất biến. Preprocessing và evaluation là offline research runner; `ClientWorker` giữ tensor local, FedAvg chỉ nhận state/count/metrics. Không có network transport hay deployment guarantee.

Transformer train-only cố định giữa original/FU/retrain. Do F có thể ảnh hưởng transformer gốc, tham chiếu retrain chỉ là retrain model với representation cố định, không phải end-to-end deletion của mọi artifact preprocessing.

Connected components gây distribution shift theo họ attack; báo cáo counts từ audit. MIA chỉ match binary label, không thể coi toàn bộ nonmembers có cùng distribution với members. Do đó MIA phải diễn giải cùng retrain/control, không làm chứng nhận privacy.

## Model và phương pháp

MLP hai hidden layer 64/32, ReLU, hai logits. Adam local lr=0.001, reset mỗi round/client; 15 rounds, một local epoch, batch 512. Tất cả client mỗi round. Fresh retrain dùng cùng initial seed, partition retained, rounds/epoch policy; minibatch count thay đổi khi xóa mẫu. Không warm-start retrain. Recovery 2 rounds, cùng seed schedule và sample-count weights đã cập nhật. No-op copy nguyên model, zero updates.

ConPFU-inspired bắt đầu original; một cycle GA/GD, SGD, regularization quanh original và clipping. KD dùng teacher sau alternating đã freeze; student original, chỉ R, CE+T² KL. Giữ after-local/after-KD/final. So sánh no-KD và retain-only KD; không incentives, DP, projection baseline hoặc last-round subtraction. Các thông số cụ thể trong config là pilot preset, chưa được tối ưu hay khẳng định tốt nhất.

## Đánh giá và gate

Threshold detection 0.5 cố định. Test dùng báo cáo cuối, không chọn hyperparameter. Utility và prediction MAE/disagreement/JS/KL tới fresh retrain trên F/R/test. Cùng request so sánh original/no-op/retain-only/retrain/local/KD/final. Pilot một seed không có CI robustness giả tạo.

MIA score cố định = negative CE, attacker biết nhãn; không train attack classifier. Threshold riêng mỗi class lấy percentile 99 từ original trên calibration nonmembers (strict `>` xử lý ties). Freeze thresholds cho mọi model. Attack evaluation dùng F hoặc retained positive controls và nonmembers test match chính xác class counts, tối đa 2.000 mỗi lớp. Retained controls loại mọi connected group chạm F; calibration/test group-disjoint. Lưu evaluation IDs. AUC và operating-point balanced accuracy, member rate, realized nonmember FPR; TPR@1% chỉ khi có >=1.000 nonmembers. Low-FPR estimate vẫn cần caution, pilot không có uncertainty claim.

Preset gate: original–retrain prediction MAE(F)>=0.001; FU gần retrain hơn; macro-F1>=0.65 và giảm không quá 0.03 so với max(original,retrain); original retained-control MIA AUC>=0.55; FU MIA không xa retrain hơn original. Một method phải đạt cả hai fraction. Đây là operational research criteria định trước, không formal deletion guarantee. Smoke luôn không đạt evidence gate.

DT chưa được triển khai. Nếu pilot đạt tiêu chí số học, vẫn cần review distribution-shift/MIA và repeated-seed xác nhận trước quyết định mở pha DT. Không tự thay method/hyperparameters sau nhìn test để buộc gate đạt.

## Chi phí

Đo `perf_counter` CPU sequential wall time và mỗi client compute time; bytes chỉ tensor payload dựa trên dtype/numel, không giả là wire bytes. Training log ghi rounds/counts/weights/steps/loss. Manifest input bất biến, runtime logs và artifact hashes tách riêng. Môi trường có torch CPU; không cài dependency ngoài.
