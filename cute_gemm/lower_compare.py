import argparse
import difflib
import json
from pathlib import Path
from typing import Any

import cutlass.cute as cute

import mma_gemm_2cta_commit_cutedsl as gemm_2cta_commit
import mma_gemm_2cta_manual_phase_mask_cutedsl as gemm_2cta_manual
import mma_gemm_2cta_pipeline_cutedsl as gemm_2cta_pipeline
from ref import make_inputs


VARIANTS: dict[str, Any] = {
    "2cta_commit": gemm_2cta_commit,
    "2cta_manual": gemm_2cta_manual,
    "2cta_pipeline": gemm_2cta_pipeline,
}


def _parse_mnk(text: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected M,N,K")
    return parts[0], parts[1], parts[2]


def _variant_output_dir(root: Path, variant: str) -> Path:
    out_dir = root / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.write_bytes(content)


def _compile_variant(
    variant: str,
    module,
    mnk: tuple[int, int, int],
    a,
    b,
    out_root: Path,
) -> dict[str, Any]:
    variant_dir = _variant_output_dir(out_root, variant)

    module.validate_mnk(mnk)
    c, a_tensor, b_tensor, c_tensor = module.prepare_cute_gemm(a, b)
    compiled = cute.compile[cute.KeepPTX(), cute.KeepCUBIN()](
        module.host_function,
        a_tensor,
        b_tensor,
        c_tensor,
    )

    mlir_text = str(compiled.ir_module)
    ptx_text = compiled.__ptx__ or ""
    cubin_bytes = compiled.__cubin__ or b""

    mlir_path = variant_dir / f"{variant}.mlir"
    ptx_path = variant_dir / f"{variant}.ptx"
    cubin_path = variant_dir / f"{variant}.cubin"

    _write_text(mlir_path, mlir_text)
    _write_text(ptx_path, ptx_text)
    _write_bytes(cubin_path, cubin_bytes)

    compiled.export_to_c(
        file_path=str(variant_dir),
        file_name=variant,
        function_prefix=variant,
    )

    info = {
        "variant": variant,
        "module_file": str(Path(module.__file__).resolve()),
        "function_name": compiled.function_name,
        "kernel_symbols": sorted(compiled.kernel_info.keys()),
        "mlir_path": str(mlir_path),
        "ptx_path": str(ptx_path),
        "cubin_path": str(cubin_path),
        "header_path": str(variant_dir / f"{variant}.h"),
        "object_path": str(variant_dir / f"{variant}.o"),
        "mlir_lines": len(mlir_text.splitlines()),
        "ptx_lines": len(ptx_text.splitlines()),
        "cubin_bytes": len(cubin_bytes),
        "mlir_text": mlir_text,
        "ptx_text": ptx_text,
    }
    return info


def _make_unified_diff(
    left_name: str,
    left_text: str,
    right_name: str,
    right_text: str,
) -> str:
    diff = difflib.unified_diff(
        left_text.splitlines(),
        right_text.splitlines(),
        fromfile=left_name,
        tofile=right_name,
        lineterm="",
    )
    return "\n".join(diff)


def _write_combined_report(
    report_path: Path,
    metadata: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# CuTeDSL Lowering Compare")
    lines.append("")
    lines.append("## Metadata")
    lines.append(json.dumps(metadata, indent=2, ensure_ascii=False))
    lines.append("")

    for result in results:
        lines.append(f"## Variant: {result['variant']}")
        lines.append(json.dumps({k: v for k, v in result.items() if not k.endswith("_text")}, indent=2, ensure_ascii=False))
        lines.append("")
        lines.append(f"### MLIR: {result['variant']}")
        lines.append("```mlir")
        lines.append(result["mlir_text"])
        lines.append("```")
        lines.append("")
        lines.append(f"### PTX: {result['variant']}")
        lines.append("```ptx")
        lines.append(result["ptx_text"])
        lines.append("```")
        lines.append("")

    if len(results) >= 2:
        for lhs, rhs in zip(results, results[1:]):
            mlir_diff = _make_unified_diff(
                f"{lhs['variant']}.mlir",
                lhs["mlir_text"],
                f"{rhs['variant']}.mlir",
                rhs["mlir_text"],
            )
            ptx_diff = _make_unified_diff(
                f"{lhs['variant']}.ptx",
                lhs["ptx_text"],
                f"{rhs['variant']}.ptx",
                rhs["ptx_text"],
            )
            lines.append(f"## Diff: {lhs['variant']} vs {rhs['variant']}")
            lines.append("")
            lines.append("### MLIR Diff")
            lines.append("```diff")
            lines.append(mlir_diff)
            lines.append("```")
            lines.append("")
            lines.append("### PTX Diff")
            lines.append("```diff")
            lines.append(ptx_diff)
            lines.append("```")
            lines.append("")

    _write_text(report_path, "\n".join(lines))


def main() -> None:
    from cuda.bindings import driver as cu_driver

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnk",
        type=_parse_mnk,
        default=(512, 512, 128),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS.keys()),
        default=["2cta_commit", "2cta_manual"],
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    cu_driver.cuInit(0)

    script_dir = Path(__file__).resolve().parent
    m, n, k = args.mnk
    default_out_dir = script_dir / "lowering_artifacts" / f"m{m}_n{n}_k{k}"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    a, b = make_inputs(args.mnk)

    results = []
    for variant in args.variants:
        result = _compile_variant(
            variant,
            VARIANTS[variant],
            args.mnk,
            a,
            b,
            out_dir,
        )
        results.append(result)

    metadata = {
        "mnk": args.mnk,
        "variants": args.variants,
        "out_dir": str(out_dir),
    }
    report_path = out_dir / "lower_compare_report.txt"
    metadata_path = out_dir / "lower_compare_metadata.json"

    _write_combined_report(report_path, metadata, results)
    _write_text(
        metadata_path,
        json.dumps(
            {
                **metadata,
                "results": [
                    {k: v for k, v in result.items() if not k.endswith("_text")}
                    for result in results
                ],
                "report_path": str(report_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
    )

    print(
        "LOWER_DONE",
        json.dumps(
            {
                "report_path": str(report_path),
                "metadata_path": str(metadata_path),
                "variants": args.variants,
                "mnk": args.mnk,
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
