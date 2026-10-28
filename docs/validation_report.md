# Báo cáo pilot FU — 2026-10-18

## Kết luận

**Đã triển khai và chạy end-to-end các pha 1–4; chưa chứng minh lợi ích riêng của FU và chưa qua gate. Không triển khai DT.** Run chính: `runs/pilot_v3_seed42`. Dữ liệu: `data/processed/connected_v3_seed42`. Số liệu từ `metrics.csv`, `gate.json`, `timings.json`, manifest và audit của đúng run này, không phải kết quả của paper.

## Dữ liệu và client

Raw 211.043 dòng, 44 cột; derived exact-dedup còn 190.474 dòng, 38 chiều. Train/validation/calibration/test: 114.284 / 28.571 / 19.048 / 28.571 dòng. Gộp connected components theo bidirectional five-tuple, raw feature signatures và encoded float32 signatures. Hai lần refinement; overlap giữa mọi cặp split bằng 0 cho group, reverse connection, raw signature và encoded vector. Component lớn nhất 38.869 dòng. 15 nhóm conflicting labels giữ nguyên.

Không suy rộng thành “không còn mọi loại leakage”: thiếu timestamp/session thật; grouping dùng covariates toàn dữ liệu; transformer cố định vẫn giữ thống kê original train. Đây là model unlearning có điều kiện trên representation cố định.

| Client | Tổng | Normal | Attack |
|---|---:|---:|---:|
| 0 | 14.465 | 8.657 | 5.808 |
| 1 | 11.129 | 11.025 | 104 |
| 2 | 472 | 159 | 313 |
| 3 | 14.714 | 1.944 | 12.770 |
| 4 | 73.504 | 749 | 72.755 |

Client 4 được chọn vì lớn nhất, theo policy định trước. Forget 10% = 7.349 dòng; 50% = 36.751 dòng; floor theo lớp, nested F10⊂F50. Target vẫn online với retained data. Row-level Dirichlet có thể chia một training component cho nhiều client, chỉ là synthetic ownership.

## Thiết lập

Windows 11, Python 3.13.2, PyTorch 2.6.0+cpu, NumPy 2.3.5, CPU hai threads. MLP 38→64→32→2; 15 rounds, một local epoch, Adam lr=0.001 reset mỗi round, batch 512. Fresh retrain cùng initialization seed và retained ownership, không warm-start. Recovery hai rounds all-clients với retained counts. FU/KD theo `configs/pilot.json`; checkpoint cả sau local/sau KD/sau recovery. Seed 42, alpha=0.5, một target; chưa có repeated-seed CI.

## Kết quả chính

MAE(F) là sai lệch xác suất tới retrain trên forget records, không phải regression/cost MAE.

| Quên | Model | Test macro-F1 | MAE(F) tới retrain | MIA AUC(F) |
|---|---|---:|---:|---:|
| 10% | Original | 0,889700 | 0,001712 | 0,667785 |
| 10% | Retrain | 0,889038 | 0 | 0,668628 |
| 10% | Retain-only | 0,889299 | 0,000711 | 0,669182 |
| 10% | FU no-KD | 0,889299 | 0,000700 | 0,669190 |
| 10% | FU retain-KD | 0,889299 | 0,000709 | 0,669212 |
| 50% | Original | 0,889700 | 0,007400 | 0,652981 |
| 50% | Retrain | 0,886574 | 0 | 0,664003 |
| 50% | Retain-only | 0,892096 | 0,003286 | 0,660010 |
| 50% | FU no-KD | 0,892096 | 0,003298 | 0,659998 |
| 50% | FU retain-KD | 0,888851 | 0,003323 | 0,660022 |

No-op bằng original chính xác cả tensor checkpoint và predictions. Metrics đầy đủ gồm PR-AUC/AP, attack precision/recall, FPR, ROC-AUC, log loss/confusion matrix, JS/KL/disagreement trên F/R/test và MIA controls trong JSON. Stage after-local/KD cũng được đánh giá.

FU no-KD giảm MAE(F) so với original khoảng 0,001012 ở 10% và 0,004103 ở 50%. Nhưng retain-only đạt gần tương đương; ở 50%, retain-only còn có MAE thấp hơn cả hai FU variant. Sau local và KD chưa recovery, MAE(F) vẫn quanh mức original (10%: 0,001777/0,001788; 50%: 0,007393/0,007437). Cải thiện quan sát chủ yếu đi kèm recovery; chưa thể quy cho GA/GD hay KD.

## Vì sao gate chưa đạt

Original retained-control MIA AUC chỉ 0,4720 (10%) và 0,4680 (50%), dưới ngưỡng 0,55 định trước. Không đảo chiều score hoặc đổi attack sau nhìn test để làm gate đạt. Forget MIA AUC của retrain vẫn ~0,66 dù F không dùng trong retraining: score có thể phản ánh mức dễ phân loại/distribution shift, không đo membership đáng tin cậy.

Controls tách group khỏi F; nonmembers match binary label nhưng không đủ attack-family/connection characteristics. Một phần F còn có predictor-equivalent retained records trong train. Các yếu tố này giới hạn khả năng nhận diện ảnh hưởng sample deletion.

Utility floor macro-F1 đạt, nhưng FPR original **27,23%**, FU no-KD 50% **25,97%** là cao cho IDS. Threshold 0,5 khóa trước pilot; không tự chỉnh bằng test. PR-AUC cao không thay kiểm tra false alarms.

Gate hiện chưa yêu cầu hơn retain-only bằng một margin; báo cáo bổ sung đối chiếu đó và không dùng việc vượt original để khẳng định lợi thế FU. `pilot_passed=false`, `dt_allowed=false`.

## Kiểm thử và artifact

- 10 tests đạt: transitive grouping, deterministic/disjoint preprocessing, train-only transforms, unknown/missing, invalid input/output, client ownership/nested F/R, sample-weighted aggregation, GA sign, fresh initialization/no-op, KD gradient/temperature và metric/MIA synthetic controls.
- Smoke v3 chạy toàn luồng trên subset khai báo rõ, một training/recovery round; không dùng làm evidence.
- Full pilot hoàn thành 16 entries (8 models/stages × 2 fractions), không bỏ case xấu.
- `verify_run.py` xác minh **45 artifacts**, manifest/data hashes, five-client ownership, holdout exclusion, nested requests, no-op tensor equality, retained weights/counts trong retraining/recovery.
- Figures từ CSV tại `results/figures/pilot_v3_seed42`, có lệnh regenerate trong README.

## Chi phí và giới hạn

Pilot runner đo 31,56 giây, không gồm Python import/startup và preprocessing. Original training 11,71 giây; fresh retrain 10%/50% 5,38/5,19 giây. Có warm-up/lazy optimizer initialization; **không dùng thời gian này để tuyên bố speedup công bằng**.

FL payload upload/download cộng tất cả comparator lần lượt 5.291.880 bytes mỗi chiều; tổng client compute FL 26,36 giây. Sequential simulation, không wire traffic/network latency; chưa gồm FU transfer, peak memory, DT overhead hoặc deployment. DT/cost comparison chưa triển khai.

## Bước tiếp theo

1. Thiết kế shadow-model hoặc learned loss/confidence MIA trên auxiliary/calibration groups tách biệt; match attack-family khi đủ support; khóa protocol trước run mới. Controls vẫn yếu thì kết luận inconclusive.
2. Chọn threshold IDS theo FPR constraint trên validation, thêm FPR gate cho nghiên cứu tiếp theo; giữ nguyên báo cáo test pilot này.
3. Ablation no-GA/retain-only với local compute tương ứng và pilot hyperparameter grid định trước; so với recovery-only hiện có. Chưa chạy full 36-case grid khi signal chưa rõ.
4. Khi FU có evidence đủ, chạy nhiều seed/target rồi mới mở DT. Không dùng DP/hybrid/incentives để che failure primary.
