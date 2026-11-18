# 提前核验第一步：白芍四图实测

运行：case_ev1_20261111_125716 / fusion_e197f99cc63b4de5a5e8b079e7c44d9f。
新case、空A缓存、原四图及1800秒预算；workflow配置与基线唯一差异是early_verification=true。

| 指标（从workflow起点计时） | 基线 | 本次 |
|---|---:|---:|
| 首次相关分割开始 | 1389.716秒 | 139.674秒 |
| 首次测量开始 | 1453.254秒 | 183.122秒 |
| 首次测量完成 | 1453.832秒 | 183.586秒 |

首次对象为view_02_01簇晶，现有质量门通过，白芍簇晶IN_RANGE；不是无关实例测量。首例采用现有YOLO高分准入，后续CNN缓存同crop为簇晶0.874697；不代表已证明准确率等价。首次分割实际送模菜单在第5轮，包含其他合法选择；模型自主选择核验。

全过程执行detect 4、classify_target 1、segment 2、measure 2、feature_scan 0；两个测量仍是同一独立特征类。
任务400.752秒收尾，status=tentative，top=白芍，formal_score=0.8823，confidence=1.0，support_count=1，review_required=true；淀粉粒/导管/纤维仍无正式支持，不能写成已排除或完整检查。

**完整耗时对照未完成。** 第12次规划收到HTTP429/vendor1113，运行时报告VLM_CREDIT_EXHAUSTED，stop_reason=runtime_request_review，文本研判回退deterministic。不能把400.752秒相对1804.8秒的下降认定为完整提速，也不能把本次较少的候选检查视为等价质量。
11次有usage的规划累计728925输入token；上下文压缩仍未实施。本次只验证了核验启动提前，后续全程对照需恢复该模型服务额度。

源码签名全程未变。回归329项（328通过、1 skip），原B59/21/125断言通过。原始工件保留于runs目录，比较证据见verification/CODE-124/online_comparison.json。
