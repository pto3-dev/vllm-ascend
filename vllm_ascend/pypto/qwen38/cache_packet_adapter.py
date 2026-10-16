# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Model-local typed GM views over one canonical native hybrid-pool argument.

PyPTO still generates every kernel and the chip orchestration. This adapter
contracts only the boundary packet and creates checked typed GM descriptors in
the generated orchestration. No cache contents are read, copied or converted;
canonical buffer identity, producer/version and device placement are preserved.
"""

import ast
import json
from math import prod
from pathlib import Path

import regex as re

HELPER = r"""
#include <initializer_list>
static Tensor native_pool_view(const Tensor& pool, uint64_t byte_offset,
                               DataType dtype, std::initializer_list<uint32_t> shape) {
    const uint64_t element = get_element_size(dtype);
    const uint64_t base_offset = pool.start_offset * get_element_size(pool.dtype);
    uint64_t count = 1;
    for (uint32_t dim : shape) {
        if (dim == 0 || count > UINT32_MAX / dim) __builtin_trap();
        count *= dim;
    }
    const uint64_t extent = count * element;
    if (!pool.is_contiguous || pool.ndims != 1 || element == 0 ||
        shape.size() > MAX_TENSOR_DIMS || byte_offset % element != 0 ||
        base_offset % element != 0 || byte_offset > pool.nbytes() ||
        extent > pool.nbytes() - byte_offset ||
        base_offset + byte_offset > pool.buffer.size ||
        extent > pool.buffer.size - base_offset - byte_offset) __builtin_trap();
    Tensor view = pool;
    view.dtype = dtype;
    view.start_offset = (base_offset + byte_offset) / element;
    view.ndims = static_cast<uint32_t>(shape.size());
    uint32_t axis = 0;
    for (uint32_t dim : shape) view.shapes[axis++] = dim;
    view.refresh_row_major_derived();
    view.is_contiguous = true;
    return view;
}
"""


def contract_cache_packet(build, typed_spec, pooled_spec):
    build = Path(build)
    layout = {
        name: (pool, offset, shape, dtype) for name, pool, offset, shape, dtype in pooled_spec.native_cache_layout
    }
    old_names = [name for name, _, _, _ in typed_spec.arguments]
    cache_indices = [i for i, name in enumerate(old_names) if name in layout]
    start, end = min(cache_indices), max(cache_indices) + 1
    if cache_indices != list(range(start, end)):
        raise ValueError("Typed native cache parameters must form one contiguous range.")
    pools = list(pooled_spec.native_cache_pools)
    delta = len(cache_indices) - len(pools)
    pool_indices = {name: start + i for i, (name, _) in enumerate(pools)}
    pool_sizes = dict(pools)
    for name, (pool, offset, shape, dtype) in layout.items():
        extent = prod(shape) * (4 if dtype == "FP32" else 2)
        if offset < 0 or offset + extent > pool_sizes[pool] or offset % (4 if dtype == "FP32" else 2):
            raise ValueError(f"Invalid native typed view: {name}")
    meta_path = build / "distributed_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["params"][start:end] = [
        dict(
            name=name + "__ssa_v0",
            shape=[2, size // 2],
            dtype="bfloat16",
            direction="InOut",  # codespell:ignore inout
        )
        for name, size in pools
    ]
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    for rank in (0, 1):
        root = build / "next_levels" / f"rank{rank}"
        cpp = root / "orchestration" / f"rank{rank}.cpp"
        source = cpp.read_text()
        declarations = "\n".join(
            f"    const Tensor& ext_{pool} = orch_args.tensor({index}).ref();" for pool, index in pool_indices.items()
        )
        source = source.replace(
            "    // External tensors",
            "    // Canonical native hybrid pools\n" + declarations + "\n    // External tensors",
            1,
        )
        pattern = (
            r"const Tensor& ext_(l\d+_(?:conv_state|recurrent_state|keys|values)) = orch_args.tensor\((\d+)\).ref\(\);"
        )

        def typed(match):
            name = match.group(1)
            pool, offset, shape, dtype = layout[name]
            type_name = "FLOAT32" if dtype == "FP32" else "BFLOAT16"
            dimensions = ", ".join(map(str, shape))
            return (
                f"const Tensor ext_{name} = native_pool_view(ext_{pool}, {offset}, "
                f"DataType::{type_name}, {{{dimensions}}});"
            )

        source, count = re.subn(pattern, typed, source)
        if count != len(layout):
            raise ValueError("Generated orchestration cache declarations changed.")

        def remap(match):
            index = int(match.group(1))
            return f"orch_args.tensor({index - delta if index >= end else index})"

        # Pool indices inserted above already name the contracted packet.
        source = re.sub(r"orch_args.tensor\((\d+)\)", remap, source)
        source = re.sub(
            r"\.expected_arg_count = (\d+)", lambda m: f".expected_arg_count = {int(m.group(1)) - delta}", source
        )
        source = source.replace('extern "C" {', HELPER + '\nextern "C" {', 1)
        cpp.write_text(source)
        config = root / "kernel_config.py"
        tree = ast.parse(config.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "ORCHESTRATION" for t in node.targets
            ):
                for key, value in zip(node.value.keys, node.value.values):
                    if isinstance(key, ast.Constant) and key.value == "signature":
                        value.elts[start:end] = [
                            ast.parse("_D.INOUT", mode="eval").body  # codespell:ignore inout
                            for _ in pools
                        ]
        config.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    host = build / "orchestration/host_orch.py"
    tree = ast.parse(host.read_text())
    removed = {}
    seen_assign = set()
    seen_arg = set()

    class Contract(ast.NodeTransformer):
        def visit_Assign(self, node):
            if isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Subscript):
                base = node.value.value
                if (
                    isinstance(base.value, ast.Name)
                    and base.value.id == "tensors"
                    and isinstance(base.slice, ast.Constant)
                ):
                    name = str(base.slice.value).split("__ssa_")[0]
                    if name in layout:
                        rank = int(node.value.slice.elts[0].value)
                        temp = node.targets[0].slice.value
                        removed[temp] = rank
                        if rank in seen_assign:
                            return None
                        seen_assign.add(rank)
                        return ast.parse(
                            "\n".join(
                                (
                                    f'tensors["native_pool_{rank}_{pool}"] = '
                                    f'tensors["{pool}__ssa_v0"][{rank},0:{size // 2}]'
                                )
                                for pool, size in pools
                            )
                        ).body
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "add_chip_tensor_arg"
            ):
                call = node.value
                temps = [
                    n.slice.value
                    for n in ast.walk(call.args[2])
                    if isinstance(n, ast.Subscript)
                    and isinstance(n.value, ast.Name)
                    and n.value.id == "tensors"
                    and isinstance(n.slice, ast.Constant)
                ]
                if temps and temps[0] in removed:
                    rank = removed[temps[0]]
                    if rank in seen_arg:
                        return None
                    seen_arg.add(rank)
                    packet = ast.unparse(call.args[0])
                    callable_name = ast.unparse(call.args[5])
                    return ast.parse(
                        "\n".join(
                            f'{packet_call}({packet}, orch._worker, tensors["native_pool_{rank}_{pool}"], '
                            f"TensorArgType.INOUT, "  # codespell:ignore inout
                            f"callables, {callable_name}, {pool_indices[pool]})"
                            for packet_call in ["add_chip_tensor_arg"]
                            for pool, _ in pools
                        )
                    ).body
                arg_index = call.args[6]
                if not isinstance(arg_index, ast.Constant):
                    raise ValueError("Expected static chip tensor argument index")
                if arg_index.value >= end:
                    call.args[6] = ast.Constant(arg_index.value - delta)
                return node
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "add_tensor"
            ):
                temps = [
                    n.slice.value
                    for n in ast.walk(node.value)
                    if isinstance(n, ast.Subscript)
                    and isinstance(n.value, ast.Name)
                    and n.value.id == "tensors"
                    and isinstance(n.slice, ast.Constant)
                ]
                if temps and temps[0] in removed:
                    rank = removed[temps[0]]
                    if rank in seen_arg:
                        return None
                    seen_arg.add(rank)
                    packet = node.value.func.value.id
                    return ast.parse(
                        "\n".join(
                            f'{packet}.add_tensor(make_tensor_arg(orch._worker,tensors["native_pool_{rank}_{pool}"]),'
                            "TensorArgType.INOUT)"  # codespell:ignore inout
                            for pool, _ in pools
                        )
                    ).body
            return self.generic_visit(node)

    tree = Contract().visit(tree)
    if seen_assign != {0, 1} or seen_arg != {0, 1}:
        raise ValueError("Generated host packet shape changed.")
    host.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    (build / "native_cache_packet.json").write_text(
        json.dumps(
            dict(
                typed_cache_parameters=len(layout),
                canonical_pool_parameters=len(pools),
                contents_copied=False,
                layout=pooled_spec.native_cache_layout,
            ),
            indent=2,
        )
    )
