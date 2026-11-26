"""融合项目的路径权威，为主进程、原 B 组件和隔离 worker 提供同一份本地路径。

activate 绑定原代码所用的 paths 和 TCM_Fusion 包名。python_environment 把定位
信息传给子进程。--install 向当前虚拟环境登记启动路径，迁移后可重新登记。
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path


PROJECT_ROOT = str(Path(__file__).resolve().parent)
COMMON_DIR = os.path.join(PROJECT_ROOT, "common")
DETECTION_DIR = os.path.join(PROJECT_ROOT, "目标检测")
DETECTION_CLI_DIR = os.path.join(DETECTION_DIR, "cli")
DETECTION_ANNOT_DIR = os.path.join(DETECTION_DIR, "annot")
DETECTION_CONFIG_DIR = os.path.join(DETECTION_DIR, "config")
DETECTION_LIB_DIR = os.path.join(DETECTION_DIR, "lib")
SEGMENTATION_DIR = os.path.join(PROJECT_ROOT, "图像分割")
SEGMENTATION_CLI_DIR = os.path.join(SEGMENTATION_DIR, "cli")
SEGMENTATION_CONFIG_DIR = os.path.join(SEGMENTATION_DIR, "config")
SEGMENTATION_LIB_DIR = os.path.join(SEGMENTATION_DIR, "lib")
GEOM_DIR = os.path.join(PROJECT_ROOT, "几何测量")
JUDGE_DIR = os.path.join(PROJECT_ROOT, "Agent判药")
JUDGE_SCRIPTS_DIR = os.path.join(JUDGE_DIR, "scripts")
MICROSCOPE_DIR = os.path.join(PROJECT_ROOT, "MICROSCOPE-main")
A_PREDICT = os.path.join(MICROSCOPE_DIR, "predict_herb_resnet18.py")
A_LEGACY_PREDICT = os.path.join(MICROSCOPE_DIR, "predictv3_resnet50.py")
A_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "分类器设计权重与CNN权重", "herb_cls_runs")
A_MODEL_RUNS = tuple(f"v2_resnet18_448_lb_e40_ls10_mix10_s{seed}"
                     for seed in (20261111, 20261112, 20261113))

VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")
PYTHON_ENTRY = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
VENV_PY = os.path.join(VENV_DIR, *PYTHON_ENTRY)
VISION_PY = os.path.join(VENV_DIR, "vision", *PYTHON_ENTRY)
YOLO_PY = os.path.join(VENV_DIR, "yolo", *PYTHON_ENTRY)
SITE_PACKAGES = sysconfig.get_path("purelib")
PTH_NAME = "tcm_project.pth"
WORKER_BOOTSTRAP_DIR = os.path.join(PROJECT_ROOT, "lib", "runtime", "worker_bootstrap")

RESULTS = os.path.join(PROJECT_ROOT, "results")
DET_RESULTS = os.path.join(RESULTS, "detection")
SEG_RESULTS = os.path.join(RESULTS, "segmentation")
UI_OUT_ROOT = os.path.join(RESULTS, "ui_out")
RUN_RESULTS = os.path.join(PROJECT_ROOT, "运行结果")
PIPELINE_ROOT = os.path.join(PROJECT_ROOT, "流水线数据")
RUNS_ROOT = os.path.join(PIPELINE_ROOT, "runs")
INPUT_ROOT = os.path.join(PIPELINE_ROOT, "input")
DATASET = os.environ.get("TCM_DATASET_ROOT", "").strip() or PROJECT_ROOT
IMAGE_SUBDIRS = (os.path.join("dataset", "显微镜像"), os.path.join("dataset", "数据量太少合集"))
DET_TRAIN_V1 = os.path.join(DATASET, "dataset", "检测训练集_v1")
CLS_TRAIN_V1 = os.path.join(DATASET, "dataset", "特征分类器_v1")

MODULE_DIRS = (PROJECT_ROOT, COMMON_DIR, DETECTION_DIR, DETECTION_LIB_DIR,
               SEGMENTATION_DIR, SEGMENTATION_LIB_DIR, GEOM_DIR, JUDGE_SCRIPTS_DIR)


def activate():
    """同一路径模块兼容原 B 的顶层导入和融合包名。"""
    package = sys.modules.get("TCM_Fusion")
    package_file = Path(PROJECT_ROOT, "__init__.py").resolve()
    if package is not None and Path(package.__file__).resolve() != package_file:
        raise RuntimeError("此解释器已经载入另一份 TCM_Fusion，请启动独立进程")
    if package is not None and sys.modules.get("paths") is sys.modules[__name__]:
        return
    local = {os.path.normcase(p) for p in MODULE_DIRS}
    sys.path[:] = [*MODULE_DIRS, *(p for p in sys.path if os.path.normcase(p) not in local)]
    sys.modules["paths"] = sys.modules[__name__]
    sys.modules["TCM_Fusion.paths"] = sys.modules[__name__]
    if package is None:
        spec = importlib.util.spec_from_file_location(
            "TCM_Fusion", package_file, submodule_search_locations=[PROJECT_ROOT],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = package
        spec.loader.exec_module(package)


def python_environment(env=None):
    """原 A/B、MCP 和请求进程从同一融合目录加载代码。"""
    return dict(os.environ if env is None else env,
                TCM_FUSION_ROOT=PROJECT_ROOT,
                PYTHONPATH=os.pathsep.join((WORKER_BOOTSTRAP_DIR, *MODULE_DIRS)))


def pth_path(site_packages=None):
    return os.path.join(site_packages or SITE_PACKAGES, PTH_NAME)


def render_pth(module_dirs=None):
    root = os.path.normpath((module_dirs or MODULE_DIRS)[0])
    # ASCII 转义让 Windows 的系统编码环境也能登记中文路径。
    return f"import sys; sys.path.insert(0, {json.dumps(root, ensure_ascii=True)}); import paths; paths.activate()\n"


def check(site_packages=None, module_dirs=None):
    directories = module_dirs or MODULE_DIRS
    for folder in directories:
        if not Path(folder).is_dir():
            print(f"缺少组件目录 {folder}")
            return 1
    target = Path(pth_path(site_packages))
    if not target.is_file() or target.read_text(encoding="ascii") != render_pth(directories):
        print(f"路径登记缺失或已过期，请重新执行 paths.py --install。{target}")
        return 1
    probe = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-c",
         "import paths, TCM_Fusion, contracts; print(paths.PROJECT_ROOT)"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=30,
    )
    if probe.returncode or os.path.normcase(probe.stdout.strip()) != os.path.normcase(PROJECT_ROOT):
        print(f"独立导入检查失败。{probe.stderr.strip()}")
        return 1
    print(f"融合路径和独立解释器导入通过。{PROJECT_ROOT}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="登记或校验融合项目的独立导入路径")
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ("--install", "--uninstall", "--check", "--show"):
        actions.add_argument(action, action="store_true")
    args = parser.parse_args(argv)
    if args.show:
        print(PROJECT_ROOT)
        return 0
    if args.check:
        return check()
    if sys.prefix == sys.base_prefix:
        parser.error("请用项目虚拟环境的解释器执行路径登记")
    target = Path(pth_path())
    if args.install:
        with target.open("w", encoding="ascii", newline="\n") as stream:
            stream.write(render_pth())
        print(f"已登记 {target}")
    else:
        target.unlink(missing_ok=True)
        print(f"已移除 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
