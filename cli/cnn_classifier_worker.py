"""受控 stage2 worker：context → 训练侧 cls_net 推理 → workflow 身份概率，从项目根以 `-m` 启动。

只做显微特征 crop 分类；不生成药材结论，不调用 GLM，不改变质量门；预处理必须复用训练侧变换，checkpoint/meta 不符即失败。

主要定义：
- main: 解析 `--context`，调用 `lib.stage2_classifier.infer`，把逐 crop 预测写到 context 指定输出路径。

关联：
- TCM_Fusion.lib.stage2_classifier: 承载 cls_net 加载、变换复用与推理实现。
- TCM_Fusion.lib.workflow_backend: 以 `-m TCM_Fusion.cli.cnn_classifier_worker` 子进程调用，供 `classify_crops_batch` 使用。
- TCM_Fusion.lib.storage: 读取 context、写结果 JSON。
"""
import argparse
from TCM_Fusion.lib.stage2_classifier import infer
from TCM_Fusion.lib.storage import read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', required=True)
    context = read_json(parser.parse_args(argv).context)
    value = infer(context)
    write_json(context['output'], value)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
