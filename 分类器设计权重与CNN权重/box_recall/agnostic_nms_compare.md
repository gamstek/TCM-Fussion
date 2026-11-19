# 类别无关评测（agnostic NMS）

gold：60 图 / 239 框；imgsz=1024，conf=0.001，NMS IoU=0.45，tile=0

| 模型 | NMS | AP50 | AP50-95 | conf .25 R/P | top-5 R | FP≤1/图 R | FP≤5/图 R |
|---|---|---|---|---|---|---|---|
| v10s_multi_classnms | class-aware | 0.3776 | 0.2606 | 0.347/0.728 | 0.293 | 0.3766 | 0.477 |
| v10s_multi_agnms | agnostic | 0.3791 | 0.2615 | 0.347/0.728 | 0.293 | 0.3766 | 0.477 |
| v10s_single_agnms | agnostic | 0.3232 | 0.2250 | 0.251/0.632 | 0.284 | 0.2971 | 0.4351 |
