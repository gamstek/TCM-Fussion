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
from pathlib import Path
import unittest

import paths


class ComponentBoundaryTests(unittest.TestCase):
    def test_no_component_cycles_or_live_imports_of_offline_and_tests(self):
        root = Path(paths.PROJECT_ROOT) / 'TCM_Fusion'
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
