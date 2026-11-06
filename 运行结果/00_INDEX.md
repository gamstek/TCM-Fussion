# 当前保留的运行参考

- [case_9cc59640](case_9cc59640/)：用户指定保留的历史五图运行，run_id=fusion_822f37d328bd47bc9f883be5a52456d6；原始295个文件哈希清理前后完全一致。五张A均cache_hit=true，不能当成本次新代码重推理证据。
- [五张原图及manifest](needle_5view_final_1f76fd81/)：只保留view_01～05.jpg和input_proof.json，供正式five-view接口复制原图。不是已有crop/mask的回放入口。

旧运行与临时实验已移至项目根archive/TCM_Fusion_CODE-066_20261106/removed/运行结果，原相对路径保留；历史修改记录中的旧链接需按该归档查阅。

最新代码的五图运行使用cli.main five-view，读取config/five_view.json规范模板。新实验生成新case，不覆盖参考。CODE-066仅执行离线回归与本地doctor dry-run，没有新付费识别。
