# application

任务入口、多图输入和准备流程。

| 文件 | 阅读指引 |
| --- | --- |
| [test_lifecycle.py](test_lifecycle.py) | 验证正式运行入口的终态和模拟边界。 |
| [test_multi_view.py](test_multi_view.py) | 多图接口契约回归：源图校验、case 目录隔离、dry-run 与真实 pipeline 指针，模型调用均为显式替身。 |

统一命令和测试输入边界见 [tests 导航](../README.md)。
