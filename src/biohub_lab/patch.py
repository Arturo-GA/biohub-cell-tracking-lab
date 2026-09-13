"""Apply a narrow, fail-closed change to the frozen inference source."""
import ast


def patch_source(source, *, candidate=False):
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == '_bi_new' for target in node.targets)]
    if len(matches) != 1:
        raise ValueError('Expected exactly one harmonic runtime patch')
    if candidate:
        node = matches[0]
        original = ast.literal_eval(node.value)
        anchor = '                harmonic_prob = 1.0 / (\n                    (1.0 - _bidirectional_weight) / forward_prob\n                    + _bidirectional_weight / reverse_prob\n                )'
        replacement = '''                from biohub_lab.fusion import division_aware_reverse_weight
                _local_reverse_weight = division_aware_reverse_weight(
                    forward_prob, p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                    base_weight=_bidirectional_weight,
                )
                harmonic_prob = 1.0 / (
                    (1.0 - _local_reverse_weight) / forward_prob
                    + _local_reverse_weight / reverse_prob
                )'''
        if original.count(anchor) != 1:
            raise ValueError('Harmonic fusion anchor changed')
        changed = original.replace(anchor, replacement, 1)
        lines = source.splitlines(keepends=True)
        lines[node.lineno - 1:node.end_lineno] = ['_bi_new = ' + repr(changed) + '\n']
        source = ''.join(lines)
    anchor = 'TEST_DIR = COMP_DIR / "test"'
    if source.count(anchor) != 1:
        raise ValueError('Data directory anchor changed')
    source = source.replace(anchor, 'TEST_DIR = Path(os.environ.get("BIOHUB_LAB_DATA_DIR", str(COMP_DIR / "test")))')
    # The upstream workers overwrite PYTHONPATH. Add only our module directory
    # to that explicit path so both one- and two-GPU runs import the same patch.
    source = source.replace('"PYTHONPATH": "src"',
        '"PYTHONPATH": "src" + os.pathsep + os.environ["BIOHUB_LAB_SRC"]')
    compile(source, '<patched-inference>', 'exec')
    return source
