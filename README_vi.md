# Gói vá FU (nhánh fu-v2)

Chép đúng vào repo, KHÔNG sửa file cũ:

    src/fu/fu_v2.py            -> D:\buget-aware\src\fu\fu_v2.py
    src/fu/rollback.py         -> D:\buget-aware\src\fu\rollback.py
    scripts/fu_study.py        -> D:\buget-aware\scripts\fu_study.py
    tests/test_rollback_fu_v2.py -> D:\buget-aware\tests\test_rollback_fu_v2.py

Chạy (thư mục output phải chưa tồn tại):

    python -m unittest discover -s tests -v
    python scripts/fu_study.py --run runs/reproduction_pilot --data data/processed/reproduction_seed42 --output runs/study_f10 --fraction 0.1
    python scripts/fu_study.py --run runs/reproduction_pilot --data data/processed/reproduction_seed42 --output runs/study_f50 --fraction 0.5

Kết quả: runs/study_f10/study.csv va study_meta.json.
Hai dong [check] dau tien phai ra ~0; neu khong, dung lai va gui cho nguoi huong dan/Claude.

Luu y: goi nay CHUA duoc chay thu voi torch (sandbox het dung luong dia); chi kiem tra cu phap.
