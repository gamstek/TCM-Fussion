"""检查组件之间的导入方向和在线边界。

读取正式 Python 源码，覆盖函数内导入和组件级依赖环。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- ComponentBoundaryTests 通过 AST 检查导入声明。

关联：
- 统一执行见 tests/README.md。
"""
import ast
from graphlib import TopologicalSorter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import paths
from TCM_Fusion.lib.runtime.storage import project_path


class ComponentBoundaryTests(unittest.TestCase):
    def test_child_loads_only_bundled_components_from_an_unrelated_directory(self):
        roots = ("paths", "contracts", "ls_common", "detect_core", "igm_segment", "geomlib",
                 "rule_engine", "TCM_Fusion")
        code = (
            "import sitecustomize, importlib.util, json; "
            f"print(json.dumps({{n: importlib.util.find_spec(n).origin for n in {roots!r}}}))"
        )
        with tempfile.TemporaryDirectory() as work:
            child = subprocess.run(
                [sys.executable, "-S", "-X", "utf8", "-c", code],
                cwd=work, env=paths.python_environment(),
                capture_output=True, text=True, encoding="utf-8", timeout=30,
            )
        self.assertEqual(child.returncode, 0, child.stderr)
        package = Path(paths.PROJECT_ROOT)
        for name, source in json.loads(child.stdout).items():
            self.assertTrue(Path(source).resolve().is_relative_to(package), name)
        self.assertEqual(project_path("TCM_Fusion/config/multi_view.json"),
                         package / "config" / "multi_view.json")

    def test_no_component_cycles_or_live_imports_of_offline_and_tests(self):
        root = Path(paths.PROJECT_ROOT)
        library = root / 'lib'
        components = {folder.name for folder in library.iterdir()
                      if folder.is_dir() and (folder / '__init__.py').is_file()}
        graph = TopologicalSorter()
        for file in library.rglob('*.py'):
            owner = file.relative_to(library).parts[0]
            if owner not in components:
                continue
            graph.add(owner)
            tree = ast.parse(file.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imports = [node.module or '']
                    self.assertEqual(node.level, 0, f'{file}:{node.lineno} 使用完整组件路径')
                elif isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                else:
                    continue
                for name in imports:
                    parts = name.split('.')
                    if parts[0] != 'TCM_Fusion':
                        continue
                    self.assertNotEqual(parts[1:2], ['tests'], f'{file}:{node.lineno}')
                    if len(parts) < 3 or parts[1] != 'lib':
                        continue
                    dependency = parts[2]
                    self.assertIn(dependency, components, f'{file}:{node.lineno}')
                    if owner != dependency:
                        self.assertNotEqual(dependency, 'offline', f'{file}:{node.lineno}')
                        graph.add(owner, dependency)
        self.assertEqual(set(graph.static_order()), components)

    def test_offline_module_dependencies_have_no_delayed_import_cycle(self):
        folder = Path(paths.PROJECT_ROOT) / 'lib/offline'
        prefix = 'TCM_Fusion.lib.offline'
        modules = {f'{prefix}.{file.stem}': file for file in folder.glob('*.py')}
        graph = TopologicalSorter()
        for name, file in modules.items():
            graph.add(name)
            tree = ast.parse(file.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imports = [node.module, *(f'{node.module}.{item.name}' for item in node.names)]
                elif isinstance(node, ast.Import):
                    imports = [item.name for item in node.names]
                else:
                    continue
                graph.add(name, *(target for target in imports if target in modules))
        self.assertEqual(set(graph.static_order()), modules.keys())
