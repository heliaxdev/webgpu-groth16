"""
Generate optimized mul_montgomery_u384 and sqr_montgomery_u384 WGSL functions.

Strategy (OPT-17 + OPT-26 + OPT-4):
- Scalar variables (t0..t31) instead of array<u32, 32> accumulator
  -> Forces GPU compiler to allocate individual registers, avoiding array spill
- Fully unrolled inner loops with literal Q_MODULUS constants
  -> Eliminates array indexing, enables immediate operand encoding
- Fully unrolled carry propagation with temp variable
  -> No loop overhead, explicit data flow helps compiler
- Dedicated squaring function with single input parameter
  -> Saves 30 registers vs mul(a,a)
"""

Q_LIMBS = [
    0x0aab, 0x1ffd, 0x1fff, 0x1dff, 0x1b9f, 0x1fff,
    0x054f, 0x1fd6, 0x0bff, 0x00f5, 0x1d89, 0x0d61,
    0x0a0f, 0x1869, 0x1d9c, 0x0257, 0x1385, 0x1c27,
    0x1dd2, 0x0ec8, 0x1acd, 0x01a5, 0x1ed9, 0x0374,
    0x1a4b, 0x1f34, 0x0e5f, 0x03d4, 0x0011, 0x000d,
]

INV_Q = 0x1ffd
N = 30  # number of limbs


def t(i):
    """Name for accumulator scalar variable."""
    return f"t{i}"


def gen_carry_propagation(indent="        "):
    """Generate fully unrolled carry propagation + shift."""
    lines = []
    lines.append(f"{indent}// Phase 4: Carry propagation + shift (fully unrolled)")
    lines.append(f"{indent}var carry: u32 = {t(0)} >> 13u;")
    lines.append(f"{indent}var v: u32;")
    for j in range(1, N):
        lines.append(f"{indent}v = {t(j)} + carry; {t(j-1)} = v & 0x1FFFu; carry = v >> 13u;")
    # Final: t[N] + carry -> t[N-1] and t[N]
    lines.append(f"{indent}v = {t(N)} + carry; {t(N-1)} = v & 0x1FFFu; {t(N)} = v >> 13u; {t(N+1)} = 0u;")
    return "\n".join(lines)


def gen_mul_montgomery():
    """Generate mul_montgomery_u384 with scalar accumulators and full unrolling."""
    lines = []
    lines.append("fn mul_montgomery_u384(a: U384, b: U384) -> U384 {")

    # Declare scalar accumulator variables
    for i in range(N + 2):
        lines.append(f"    var {t(i)}: u32 = 0u;")
    lines.append("")

    lines.append(f"    for (var i: u32 = 0u; i < {N}u; i = i + 1u) {{")
    lines.append("        let ai = a.limbs[i];")
    lines.append("")

    # Phase 1: Multiply (unrolled)
    lines.append("        // Phase 1: Multiply — a[i] * b[j] (unrolled)")
    for j in range(N):
        lines.append(f"        {t(j)} = {t(j)} + ai * b.limbs[{j}u];")
    lines.append("")

    # Phase 2: Compute m
    lines.append(f"        // Phase 2: m = t0 * (-q^{{-1}}) mod 2^13")
    lines.append(f"        let m = ({t(0)} * {hex(INV_Q)}u) & 0x1FFFu;")
    lines.append("")

    # Phase 3: Reduce (unrolled with literal Q constants)
    lines.append("        // Phase 3: Reduce — m * q[j] (unrolled, literal constants)")
    for j in range(N):
        lines.append(f"        {t(j)} = {t(j)} + m * {hex(Q_LIMBS[j])}u;")
    lines.append("")

    # Phase 4: Carry propagation
    lines.append(gen_carry_propagation())

    lines.append("    }")
    lines.append("")

    # Result extraction (unrolled)
    lines.append("    var result: U384;")
    for i in range(N):
        lines.append(f"    result.limbs[{i}u] = {t(i)};")
    lines.append("")

    # Conditional subtraction
    lines.append("    // Conditional subtraction if result >= q")
    lines.append("    var is_gte = true;")
    lines.append(f"    if {t(N)} > 0u {{")
    lines.append("        is_gte = true;")
    lines.append("    } else {")
    lines.append(f"        for (var i: u32 = {N-1}u; i < {N}u; i = i - 1u) {{")
    lines.append("            if result.limbs[i] > Q_MODULUS[i] { break; }")
    lines.append("            if result.limbs[i] < Q_MODULUS[i] { is_gte = false; break; }")
    lines.append("            if i == 0u { break; }")
    lines.append("        }")
    lines.append("    }")
    lines.append("    if is_gte {")
    lines.append("        result = sub_u384(result, U384(Q_MODULUS));")
    lines.append("    }")
    lines.append("    return result;")
    lines.append("}")
    return "\n".join(lines)


def gen_sqr_montgomery():
    """Generate sqr_montgomery_u384 — dedicated squaring with scalar accumulators."""
    lines = []
    lines.append("fn sqr_montgomery_u384(a: U384) -> U384 {")

    # Declare scalar accumulator variables
    for i in range(N + 2):
        lines.append(f"    var {t(i)}: u32 = 0u;")
    lines.append("")

    lines.append(f"    for (var i: u32 = 0u; i < {N}u; i = i + 1u) {{")
    lines.append("        let ai = a.limbs[i];")
    lines.append("")

    # Phase 1: Squaring (unrolled, reuses a)
    lines.append("        // Phase 1: Multiply — a[i] * a[j] (squaring)")
    for j in range(N):
        lines.append(f"        {t(j)} = {t(j)} + ai * a.limbs[{j}u];")
    lines.append("")

    # Phase 2: Compute m
    lines.append(f"        let m = ({t(0)} * {hex(INV_Q)}u) & 0x1FFFu;")
    lines.append("")

    # Phase 3: Reduce (unrolled with literal Q constants)
    lines.append("        // Phase 3: Reduce (literal Q constants)")
    for j in range(N):
        lines.append(f"        {t(j)} = {t(j)} + m * {hex(Q_LIMBS[j])}u;")
    lines.append("")

    # Phase 4: Carry propagation
    lines.append(gen_carry_propagation())

    lines.append("    }")
    lines.append("")

    # Result extraction (unrolled)
    lines.append("    var result: U384;")
    for i in range(N):
        lines.append(f"    result.limbs[{i}u] = {t(i)};")
    lines.append("")

    # Conditional subtraction
    lines.append("    var is_gte = true;")
    lines.append(f"    if {t(N)} > 0u {{")
    lines.append("        is_gte = true;")
    lines.append("    } else {")
    lines.append(f"        for (var i: u32 = {N-1}u; i < {N}u; i = i - 1u) {{")
    lines.append("            if result.limbs[i] > Q_MODULUS[i] { break; }")
    lines.append("            if result.limbs[i] < Q_MODULUS[i] { is_gte = false; break; }")
    lines.append("            if i == 0u { break; }")
    lines.append("        }")
    lines.append("    }")
    lines.append("    if is_gte {")
    lines.append("        result = sub_u384(result, U384(Q_MODULUS));")
    lines.append("    }")
    lines.append("    return result;")
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("// ============================================================================")
    print("// BASE FIELD MONTGOMERY MULTIPLICATION (F_q) — Lazy CIOS with 13-bit limbs")
    print("//")
    print("// Scalar accumulator variables (OPT-17), literal Q constants (OPT-26).")
    print("// Eliminates array spilling and indexing overhead.")
    print("// Generated by scripts/gen_fp_unrolled.py")
    print("// ============================================================================")
    print()
    print(gen_mul_montgomery())
    print()
    print("// ============================================================================")
    print("// BASE FIELD MONTGOMERY SQUARING (F_q) — Dedicated squaring (OPT-4)")
    print("//")
    print("// Scalar accumulators + single input = minimal register pressure.")
    print("// Generated by scripts/gen_fp_unrolled.py")
    print("// ============================================================================")
    print()
    print(gen_sqr_montgomery())
