# 当前保留的运行参考

- [case_9cc59640](case_9cc59640/)：用户指定保留的历史五图运行，run_id=`fusion_822f37d328bd47bc9f883be5a52456d6`；原始文件哈希清理前后一致。该次五张 A 均 `cache_hit=true`，不是当前代码重新推理的证据。
- 示例输入副本：[龙胆_final_test](龙胆_final_test/) 内含 `view_01`～`05.jpg` 和 `input_proof.json`；它同时是一次独立运行，仅作输入格式示例。原始龙胆源图若已归档，请从归档或数据集重新准备，不要修改历史 case。
- [龙胆_final_test](龙胆_final_test/)：当前代码的独立五图运行；使用 `multi-view`、空 A 缓存。
- [麦冬针晶](麦冬针晶/)：麦冬针晶 4–8 五图运行；输入 manifest 为 `麦冬针晶_manifest.json`。
- [case_af4149f1](case_af4149f1/)：早期独立多图运行，用于对照；其中 classify/segment/measure 为 0，不能代表加入身份门后的行为。

## 当前命令

```powershell
.venv\Scripts\python.exe -X utf8 -m TCM_Fusion.cli.main multi-view `
  --source-dir <含input_proof.json的目录> `
  --a-python D:/Anaconda/python.exe `
  --json
```

`five-view` 仍为兼容别名，但新运行统一使用 `multi-view` 和 `config/multi_view.json`。新实验生成新 case，不覆盖历史参考。

旧运行与临时实验已移至项目根 `archive/TCM_Fusion_CODE-066_20261106/removed/运行结果`，原相对路径保留；历史修改记录中的旧链接需按归档查阅。

[返回 README](../README.md) · [多图接口](../多图测试运行方案.md) · [修改记录](../修改记录.md)
