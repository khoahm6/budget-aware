# Kế hoạch triển khai theo pha

> Cập nhật: đã triển khai/chạy pha 1–4 theo `experiment_protocol.md`. Pilot v3 hoàn tất nhưng evidence gate không đạt vì MIA controls yếu và chưa có lợi thế rõ so với retain-only. DT chưa mở. Xem `validation_report.md`; nội dung dưới là kế hoạch ban đầu.

## Quyết định và giới hạn

Ưu tiên `conpfu_inspired`, binary ToN-IoT, 5 client, MLP hai hidden layer, sample-level deletion tại client online. Người dùng xác nhận không có code cũ và yêu cầu preprocessing theo một trong hai paper. Chọn ConPFU làm nguồn thiết kế FU/Dirichlet; các chi tiết preprocessing không được paper quy định sẽ gắn nhãn adaptation. Không triển khai full DT trước khi pilot FU được kiểm chứng.

## Pha 0 — Audit và protocol

Đầu ra: `implementation_audit.md`, tài liệu này, inventory/hash/schema và đối chiếu paper. Chốt những gì có/thiếu, policy preprocessing, ranh giới offline audit/client/server. Hoàn tất trước sửa code. Raw CSV bất biến. Phiên bản source hiện chưa có Git: ghi source hash thay cho commit, không tạo revision giả.

## Pha 1 — Preprocessing và yêu cầu quên

Đợt đầu hiện được phép thực hiện: `src/data/prepare.py`, config JSON, tests và một smoke/full-data preprocessing run. Không cần torch/sklearn cho bước này.

- Row ID = hash nguồn + chỉ số dòng gốc; lưu mapping duplicate về dòng đại diện. Primary giữ lần đầu của exact duplicate, chỉ trên derived data; không sửa raw.
- Feature ban đầu: proto, service, conn_state; duration, src/dst bytes, missed_bytes, src/dst pkts, src/dst IP bytes. Loại IP/port khỏi predictors nhưng giữ làm grouping metadata; loại label/type và các DNS/SSL/HTTP/free-text fields khỏi X. Đây là bộ feature bảo thủ phục vụ pilot, không phải feature list của paper; cần ablation trên validation nếu mở rộng.
- Group có hướng theo 5-tuple, shuffle group bằng seed; tỷ lệ mục tiêu train/validation/calibration/test = 60/15/10/15. Tỷ lệ theo số nhóm, báo cáo tỷ lệ dòng thực tế; yêu cầu mỗi split có hai lớp. Không có time split vì thiếu timestamp. Audit reverse-endpoint overlap để thấy giới hạn của grouping có hướng.
- Median imputation và standard scaling cho numeric; one-hot categorical với missing/unknown riêng. Chỉ fit trên train; đóng băng schema/transform ở mọi baseline, ghi rõ preprocessing vẫn phụ thuộc original train.
- Đếm feature-signature duplicates/conflicts và cross-split overlap trước transform. Giữ conflicting labels nhưng ghi nhận; overlap là cảnh báo cần review, không tuyên bố leakage-free. Nếu overlap nghiêm trọng, thử nối nhóm 5-tuple và signature thành connected components rồi đánh giá split feasibility trước khi huấn luyện; không lén lọc test.
- Bước kế tiếp: Dirichlet theo label, alpha=0.5 pilot, 5 client chỉ từ train; lưu counts/labels/group composition và hash; reject empty/single-class clients. Train groups có thể trải nhiều simulated clients trong row-level Dirichlet: phải ghi rõ giả định synthetic ownership và kiểm tra overlap.
- Forget requests stratified, nested 10%/50% cho pilot; lưu ID/hash/count. Tests F thuộc một client train, F/R rời nhau và không đụng validation/calibration/test. Định nghĩa rounding trước, không fallback.

Gate: preprocessing deterministic, no cross-split group overlap, transform train-only, no label/type in X, finite fixed-dimension matrices; mọi residual signature overlap có quyết định protocol trước training. Partition/request tests hoàn tất mới coi toàn Pha 1 đạt.

## Pha 2 — Training core và retraining reference

Tạo một implementation chung cho MLP, local training, weighted FedAvg và evaluation. Nếu sau này cung cấp code cũ, đánh giá và thay bằng adapter trước khi nhân đôi logic.

Đầu ra: `src/fl/model.py`, `fedavg_core.py`, `train_fedavg.py`, `src/eval/retrain_reference.py`, config FL, checkpoint/manifest. Width MLP, optimizer, batch size, rounds/local epochs được chọn trên pilot validation rồi khóa. Có centralized/local-only sanity baselines theo đề cương.

M_original train trên D; M_retrain fresh initialization trên D\\F, giữ client assignment và cùng rounds/local epochs/selection policy. Cặp seed và client schedule được lưu, không ép minibatch giống nhau khi cardinality thay đổi. M_noop có zero FU/recovery; M_retain_only dùng retained data từ original, cùng recovery schedule. Ghi tổng gradient steps thực tế để diễn giải budget fairness.

Gate: unit test sample weights kể cả deletion; checkpoint/schema tests; no-op equivalence; fresh init và F exclusion; baseline hữu ích và non-trivial. Đo original–retrain separation, không chọn deletion theo test để tạo chênh lệch.

## Pha 3 — ConPFU-inspired và recovery chung

`src/fu/conpfu_inspired.py`: bắt đầu từ θ_original; loss forget `-CE(F)+lambda*||theta-theta_init||²`, clipping; xen kẽ descent CE(R). Lưu gradient norms/loss và checkpoint sau alternating.

KD: frozen teacher = sau alternating; student initialized từ θ_original; soft KL có T² + hard CE. Primary pilot so sánh no-KD và retain-only KD; full-local KD chỉ là ablation riêng, F chỉ được dùng trong local operation. Hai-logit softmax phù hợp binary CE/KL; test teacher không nhận gradient và student được cập nhật.

`src/fu/recovery.py`: fixed all_clients recovery cho pilot, mọi client dùng retained data; target vẫn tham gia với R. Áp dụng local FU model làm recovery start, không ngầm blend với original. Lưu after-local, after-KD, after-recovery riêng. Last-round subtraction không nằm trong bản tối thiểu.

Gate: loss-sign test, bounded/numerically stable updates, no-op, teacher freeze, F không xuất hiện trong recovery, manifest liên kết đúng original/request/checkpoint.

## Pha 4 — Pilot và kiểm chứng FU, chưa có DT

Pilot: một seed, alpha=0.5, một target chọn bằng metadata, nested requests 10% và 50%, all-client recovery với rounds cố định. Chạy original, no-op, retain-only, fresh retrain, FU no-KD và FU retain-only KD. Reuse original cho cùng split/partition; retrain riêng từng request.

Utility: macro-F1, attack precision/recall, AP (ghi rõ định nghĩa PR-AUC), FPR, ROC-AUC, log loss và confusion matrix. Threshold chỉ fit validation/calibration.

Verification: prediction MAE, disagreement, JS/KL trên F, fixed retained evaluation subset và test; lấy retrain làm reference. Báo cáo absolute distance và signed improvement so với original/no-op/retain-only, cả trước và sau recovery. Không chỉ dùng parameter distance.

MIA ban đầu: loss/confidence attack, fixed operating point fit trên các retained-member calibration IDs và matched non-members, không dùng F hoặc audit test để tune. Member controls tách khỏi evaluation members; non-member calibration/evaluation group-disjoint. So sánh cùng attack protocol trên original/retrain/FU, báo cáo AUC/balanced accuracy/TPR@1% nếu đủ mẫu. Attack yếu ⇒ inconclusive. Research-only evaluator có quyền đọc F đã đóng băng; server/recovery không có quyền dùng F.

Trước chạy khóa utility floor, minimum identifiable separation và tiêu chí pilot bằng validation/pilot protocol; không chọn ngưỡng sau nhìn test. Pilot một seed chỉ kiểm tra tính vận hành và phát hiện failure, chưa hỗ trợ robustness claims. Gate FU: không leakage chưa giải quyết; đầy đủ controls; metric thật; có bằng chứng giảm khoảng cách tới retrain mà không phá utility hoặc báo inconclusive/failure và điều tra. Không mở DT nếu gate chưa đạt.

## Pha 5 — Robustness FU và baseline tùy chọn

Sau pilot, 3 seeds × 2 alpha (0.1/0.5) × 2 target × 3 fractions (10/30/50%) = 36 cases/method. Không full Cartesian hyperparameter grid. Paired differences/CI theo seed/request, không xem các row là seed độc lập; ba seed cho CI còn yếu phải nêu rõ. `coverage.csv` ghi cả failures và cases invalid.

`fedkd_inspired` tùy chọn: projected ascent, original-center adaptation, auxiliary data tách biệt nếu server KD; cùng recovery với primary; không DP claim. Hybrid chỉ sau khi có failure mode chứng minh cần thiết. Ablation no-GA, no-retain alternation, no-KD, no-recovery được chọn gọn. Không incentives/client deletion/class deletion/continuous deletion.

## Pha 6–7 — DT, chỉ sau FU gate

Triển khai timestamped state, đo wall time/compute sum/payload và wire bytes phân biệt, memory và DT overhead; cost model fit trên past runs, online feedback. Planner subset không rỗng, budget time/bytes và representation constraints; trả `NO_FEASIBLE_PLAN` khi không khả thi.

So sánh target_only/all_clients/random_k/greedy/static/adaptive với cùng FU intermediate và recovery rounds. Budgets tight/moderate/loose từ measured baselines; stable/load/bandwidth/stale-state scenarios. Không dùng test hoặc chi phí tương lai để chọn plan. Oracle chỉ evaluation upper bound. Gate: measured budget violations, cost error, utility, similarity và overhead thực.

## Pha 8 — Tích hợp và báo cáo

FastAPI/Docker Compose proof of concept sau experimental core; registry, request validation, trạng thái tác vụ, reproducibility. Báo cáo tiếng Việt và figures từ actual run IDs, failures/coverage đầy đủ. Tách implemented / tested / experiment completed / empirically supported. Không gọi approximate FU là certified deletion hay DP.

## Provenance và kiểm thử xuyên suốt

Immutable input manifest: dataset/source/config/schema/partition/request/checkpoint hashes, seed, model/optimizer config, software/hardware, revision hoặc source hashes. Mutable metrics/timing/error logs tách riêng; không overwrite run cũ. Mỗi pha: unit tests cần thiết, cheap smoke, rồi mới run lớn. Báo lỗi schema/NaN/invalid split/dependency thiếu thay vì fallback.

Các tên module/CLI training/FU/DT ở trên là **đề xuất**, chưa có lệnh chạy hợp lệ cho đến khi được triển khai. Lệnh preprocessing thực tế sẽ ghi trong README sau khi chạy kiểm thử.
