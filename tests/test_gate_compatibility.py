"""Reproducible separate-process baseline comparisons, never a branch checkout."""
import json
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_disabled_controls_preserve_common_numerical_kernel_asts():
    old = subprocess.check_output(['git','show',
        '7648185:src/nh_flood_2d/core/solver_compact.py'], text=True)
    new = (Path(__file__).parents[1]/'src/nh_flood_2d/core/solver_compact.py').read_text()
    class WithoutGateDiagnostics(ast.NodeTransformer):
        def visit_If(self,node):
            if ast.dump(node.test) == ast.dump(ast.parse('ti.static(gate_enabled)',mode='eval').body):
                return None
            return self.generic_visit(node)
    def kernels(source):
        tree = WithoutGateDiagnostics().visit(ast.parse(source))
        return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)}
    before, after = kernels(old), kernels(new)
    for name in ('prepare_step','finish_step','initialize','apply_initial_levels',
                 'tick','compute_cfl','initial_diagnostics','reset_extrema',
                 'sample_stations','update_pressure'):
        assert before[name] == after[name], name


@pytest.mark.parametrize('case', ['small', 'hk'])
def test_7648185_no_gate_and_disabled_are_bitwise_equal(tmp_path, case):
    command = [sys.executable, '-B', str(Path(__file__).with_name('gate_compatibility.py')),
               '--case', case, '--output', str(tmp_path/'comparison')]
    if case == 'hk':
        domain = os.environ.get('NH_GATE_HK_DOMAIN')
        force = os.environ.get('NH_GATE_HK_FORCE')
        if not domain or not force:
            pytest.skip('Set NH_GATE_HK_DOMAIN and NH_GATE_HK_FORCE for the 1 h CPU smoke comparison')
        command += ['--domain', domain, '--force', force]
    subprocess.run(command, check=True)
    result = json.loads((tmp_path/'comparison/comparison.json').read_text())
    assert result['bitwise_equal']
    assert result['results']['7648185']['nonzero_discharge_edges'] > 0
