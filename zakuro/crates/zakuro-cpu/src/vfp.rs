//! VFPv2 floating point.

use zakuro_common::bits::{bit, bits};

use crate::{Bus, Cpu, Exit};

/// bits 31:28 of FPSCR, the comparison result flags.
const FPSCR_FLAG_MASK: u32 = 0xF000_0000;
/// flush-to-zero.
const FPSCR_FZ: u32 = 1 << 24;
/// default NaN mode.
const FPSCR_DN: u32 = 1 << 25;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Vfp {
    /// s0..s31 as raw bits. dN occupies s(2N) (low word) and s(2N+1).
    pub regs: [u32; 32],
    pub fpscr: u32,
    /// FPEXC. Bit 30 is EN, the kernel sets it before a thread uses VFP.
    pub fpexc: u32,
}

impl Default for Vfp {
    fn default() -> Self {
        Self::new()
    }
}

impl Vfp {
    pub fn new() -> Vfp {
        Vfp {
            regs: [0; 32],
            fpscr: 0,
            fpexc: 1 << 30,
        }
    }

    #[inline(always)]
    pub fn get_f32(&self, index: usize) -> f32 {
        f32::from_bits(self.regs[index & 31])
    }

    #[inline(always)]
    pub fn set_f32(&mut self, index: usize, value: f32) {
        self.regs[index & 31] = self.flush(value).to_bits();
    }

    #[inline(always)]
    pub fn get_f64(&self, index: usize) -> f64 {
        let lo = self.regs[(index * 2) & 31] as u64;
        let hi = self.regs[(index * 2 + 1) & 31] as u64;
        f64::from_bits((hi << 32) | lo)
    }

    #[inline(always)]
    pub fn set_f64(&mut self, index: usize, value: f64) {
        let value = self.flush_f64(value);
        let bits = value.to_bits();
        self.regs[(index * 2) & 31] = bits as u32;
        self.regs[(index * 2 + 1) & 31] = (bits >> 32) as u32;
    }

    /// applies flush-to-zero if the guest enabled it.
    #[inline(always)]
    fn flush(&self, value: f32) -> f32 {
        if self.fpscr & FPSCR_FZ != 0 && value != 0.0 && value.is_subnormal() {
            if value.is_sign_negative() {
                -0.0
            } else {
                0.0
            }
        } else {
            value
        }
    }

    #[inline(always)]
    fn flush_f64(&self, value: f64) -> f64 {
        if self.fpscr & FPSCR_FZ != 0 && value != 0.0 && value.is_subnormal() {
            if value.is_sign_negative() {
                -0.0
            } else {
                0.0
            }
        } else {
            value
        }
    }

    #[inline(always)]
    fn read_f32(&self, index: usize) -> f32 {
        self.flush(self.get_f32(index))
    }

    #[inline(always)]
    fn read_f64(&self, index: usize) -> f64 {
        self.flush_f64(self.get_f64(index))
    }

    pub fn is_enabled(&self) -> bool {
        self.fpexc & (1 << 30) != 0
    }
}

// ---------------------------------------------------------------------------
// Register number decoding
// ---------------------------------------------------------------------------

#[inline(always)]
fn sreg(vd: u32, extra: bool) -> usize {
    ((vd << 1) | extra as u32) as usize
}

#[inline(always)]
fn dreg(vd: u32, extra: bool) -> usize {
    (((extra as u32) << 4) | vd) as usize
}

// ---------------------------------------------------------------------------
// Data processing (CDP space, coprocessors 10 and 11)
// ---------------------------------------------------------------------------

pub fn data_processing(cpu: &mut Cpu, op: u32) -> Option<Exit> {
    let coproc = bits(op, 8, 11);
    if coproc != 10 && coproc != 11 {
        return Some(Exit::Undefined {
            pc: cpu.current_pc(),
            opcode: op,
        });
    }
    let double = coproc == 11;

    let d_bit = bit(op, 22);
    let n_bit = bit(op, 7);
    let m_bit = bit(op, 5);
    let vd = bits(op, 12, 15);
    let vn = bits(op, 16, 19);
    let vm = op & 0xF;

    let (rd, rn, rm) = if double {
        (dreg(vd, d_bit), dreg(vn, n_bit), dreg(vm, m_bit))
    } else {
        (sreg(vd, d_bit), sreg(vn, n_bit), sreg(vm, m_bit))
    };

    // bit 22 inside this field is the D register-number bit, not part of the
    // opcode, so it has to be masked out before comparing.
    let opc1 = bits(op, 20, 23) & 0b1011;
    // opc3's low bit selects between the pairs (VADD/VSUB, VMUL/VNMUL, ...).
    let negate = bit(op, 6);

    let operation = match (opc1, negate) {
        (0b0000, false) => Arithmetic::MultiplyAccumulate,
        (0b0000, true) => Arithmetic::MultiplySubtract,
        (0b0001, false) => Arithmetic::NegateMultiplySubtract,
        (0b0001, true) => Arithmetic::NegateMultiplyAccumulate,
        (0b0010, false) => Arithmetic::Multiply,
        (0b0010, true) => Arithmetic::NegateMultiply,
        (0b0011, false) => Arithmetic::Add,
        (0b0011, true) => Arithmetic::Subtract,
        (0b1000, false) => Arithmetic::Divide,
        // the extension group, unary operations, comparisons and conversions.
        (0b1011, true) => return extension(cpu, op, double, rd, rm),
        _ => {
            return Some(Exit::Undefined {
                pc: cpu.current_pc(),
                opcode: op,
            })
        }
    };

    for (d, n, m) in Vector::new(cpu.vfp.fpscr, double, rd).registers(rd, rn, rm) {
        // single-precision operations are computed in f32 rather than promoted
        // to f64, so the result is rounded exactly once as hardware does.
        let rmode = bits(cpu.vfp.fpscr, 22, 23);
        let fz = cpu.vfp.fpscr & FPSCR_FZ != 0;
        if double {
            let (a, b, acc) = (
                cpu.vfp.read_f64(n),
                cpu.vfp.read_f64(m),
                cpu.vfp.read_f64(d),
            );
            let result = operation.apply_f64(a, b, acc, rmode, fz, |v| cpu.vfp.flush_f64(v));
            if result.is_nan() && !(a.is_nan() || b.is_nan() || acc.is_nan()) {
                trace_nan(cpu, op, operation, [a, b, acc]);
            }
            cpu.vfp.set_f64(d, result);
        } else {
            let (a, b, acc) = (
                cpu.vfp.read_f32(n),
                cpu.vfp.read_f32(m),
                cpu.vfp.read_f32(d),
            );
            let result = operation.apply_f32(a, b, acc, rmode, fz, |v| cpu.vfp.flush(v));
            if result.is_nan() && !(a.is_nan() || b.is_nan() || acc.is_nan()) {
                trace_nan(cpu, op, operation, [a as f64, b as f64, acc as f64]);
            }
            cpu.vfp.set_f32(d, result);
        }
    }
    None
}

/// debugging aid, RUST_LOG=zakuro_cpu::vfp::nan=trace reports operations
/// that turn ordinary numbers into NaN.
#[cold]
fn trace_nan(cpu: &Cpu, op: u32, operation: Arithmetic, operands: [f64; 3]) {
    log::trace!(
        target: "zakuro_cpu::vfp::nan",
        "NaN from {operation:?} at 0x{:08X} (0x{op:08X}): operands {operands:?}, fpscr 0x{:08X}, lr 0x{:08X}",
        cpu.current_pc(),
        cpu.vfp.fpscr,
        cpu.regs[14],
    );
}

/// the two-operand data-processing instructions.
#[derive(Debug, Clone, Copy)]
enum Arithmetic {
    MultiplyAccumulate,
    MultiplySubtract,
    NegateMultiplyAccumulate,
    NegateMultiplySubtract,
    Multiply,
    NegateMultiply,
    Add,
    Subtract,
    Divide,
}

impl Arithmetic {
    /// a and b are the operands, acc the destination's old value.
    fn apply<T>(self, a: T, b: T, acc: T, flush: impl Fn(T) -> T) -> T
    where
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + Copy,
    {
        match self {
            Arithmetic::MultiplyAccumulate => acc + flush(a * b),
            Arithmetic::MultiplySubtract => acc - flush(a * b),
            Arithmetic::NegateMultiplyAccumulate => -flush(a * b) - acc,
            Arithmetic::NegateMultiplySubtract => flush(a * b) - acc,
            Arithmetic::Multiply => a * b,
            Arithmetic::NegateMultiply => -flush(a * b),
            Arithmetic::Add => a + b,
            Arithmetic::Subtract => a - b,
            Arithmetic::Divide => a / b,
        }
    }

    fn apply_f32(self, a: f32, b: f32, acc: f32, rmode: u32, fz: bool, flush: impl Fn(f32) -> f32) -> f32 {
        if rmode == 0 && !fz {
            return self.apply(a, b, acc, flush);
        }
        match self {
            Arithmetic::MultiplyAccumulate => {
                add_f32_rmode(acc, flush(mul_f32_rmode(a, b, rmode, fz)), rmode, fz)
            }
            Arithmetic::MultiplySubtract => {
                sub_f32_rmode(acc, flush(mul_f32_rmode(a, b, rmode, fz)), rmode, fz)
            }
            Arithmetic::NegateMultiplyAccumulate => {
                sub_f32_rmode(-flush(mul_f32_rmode(a, b, rmode, fz)), acc, rmode, fz)
            }
            Arithmetic::NegateMultiplySubtract => {
                sub_f32_rmode(flush(mul_f32_rmode(a, b, rmode, fz)), acc, rmode, fz)
            }
            Arithmetic::Multiply => mul_f32_rmode(a, b, rmode, fz),
            Arithmetic::NegateMultiply => -flush(mul_f32_rmode(a, b, rmode, fz)),
            Arithmetic::Add => add_f32_rmode(a, b, rmode, fz),
            Arithmetic::Subtract => sub_f32_rmode(a, b, rmode, fz),
            Arithmetic::Divide => div_f32_rmode(a, b, rmode, fz),
        }
    }

    fn apply_f64(self, a: f64, b: f64, acc: f64, rmode: u32, fz: bool, flush: impl Fn(f64) -> f64) -> f64 {
        if rmode == 0 && !fz {
            return self.apply(a, b, acc, flush);
        }
        match self {
            Arithmetic::MultiplyAccumulate => {
                add_f64_rmode(acc, flush(mul_f64_rmode(a, b, rmode, fz)), rmode, fz)
            }
            Arithmetic::MultiplySubtract => {
                sub_f64_rmode(acc, flush(mul_f64_rmode(a, b, rmode, fz)), rmode, fz)
            }
            Arithmetic::NegateMultiplyAccumulate => {
                sub_f64_rmode(-flush(mul_f64_rmode(a, b, rmode, fz)), acc, rmode, fz)
            }
            Arithmetic::NegateMultiplySubtract => {
                sub_f64_rmode(flush(mul_f64_rmode(a, b, rmode, fz)), acc, rmode, fz)
            }
            Arithmetic::Multiply => mul_f64_rmode(a, b, rmode, fz),
            Arithmetic::NegateMultiply => -flush(mul_f64_rmode(a, b, rmode, fz)),
            Arithmetic::Add => add_f64_rmode(a, b, rmode, fz),
            Arithmetic::Subtract => sub_f64_rmode(a, b, rmode, fz),
            Arithmetic::Divide => div_f64_rmode(a, b, rmode, fz),
        }
    }
}

/// VFPv2's short vectors.
// of course the SDK does its matrix math in vector mode. of course it does
struct Vector {
    iterations: usize,
    stride: usize,
    bank: usize,
}

impl Vector {
    fn new(fpscr: u32, double: bool, destination: usize) -> Vector {
        let bank = if double { 4 } else { 8 };
        let length = bits(fpscr, 16, 18) as usize + 1;
        Vector {
            iterations: if destination < bank { 1 } else { length },
            stride: if bits(fpscr, 20, 21) == 0b11 { 2 } else { 1 },
            bank,
        }
    }

    fn step(&self, register: usize, iteration: usize) -> usize {
        let base = register & !(self.bank - 1);
        base | ((register + iteration * self.stride) & (self.bank - 1))
    }

    /// the (d, n, m) registers of each iteration.
    fn registers(&self, d: usize, n: usize, m: usize) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        let scalar_m = m < self.bank;
        (0..self.iterations).map(move |i| {
            let m = if scalar_m { m } else { self.step(m, i) };
            (self.step(d, i), self.step(n, i), m)
        })
    }
}

fn extension(cpu: &mut Cpu, op: u32, double: bool, rd: usize, rm: usize) -> Option<Exit> {
    let opc2 = bits(op, 16, 19);
    let top = bit(op, 7);

    match (opc2, top) {
        // VMOV (register), VABS, VNEG and VSQRT take part in short vectors
        // like the arithmetic does.
        (0b0000, _) | (0b0001, _) => {
            let registers: Vec<(usize, usize)> = Vector::new(cpu.vfp.fpscr, double, rd)
                .registers(rd, rd, rm)
                .map(|(d, _, m)| (d, m))
                .collect();
            for (d, m) in registers {
                monadic(cpu, double, opc2 == 0b0001, top, d, m);
            }
        }
        // VCMP / VCMPE against a register
        (0b0100, _) => compare(cpu, double, rd, Some(rm)),
        // VCMP / VCMPE against zero
        (0b0101, _) => compare(cpu, double, rd, None),
        // VCVT between single and double
        (0b0111, true) => {
            let rmode = bits(cpu.vfp.fpscr, 22, 23);
            let fz = cpu.vfp.fpscr & FPSCR_FZ != 0;
            if double {
                // source is double, destination single.
                let v = f64_to_f32_rmode(cpu.vfp.read_f64(rm), rmode, fz);
                let sd = sreg(bits(op, 12, 15), bit(op, 22));
                cpu.vfp.set_f32(sd, v);
            } else {
                let sm = sreg(op & 0xF, bit(op, 5));
                let v = cpu.vfp.read_f32(sm) as f64;
                let dd = dreg(bits(op, 12, 15), bit(op, 22));
                cpu.vfp.set_f64(dd, v);
            }
        }
        // VCVT from integer
        (0b1000, _) => {
            let sm = sreg(op & 0xF, bit(op, 5));
            let raw = cpu.vfp.regs[sm];
            // bit 7 selects signed (1) or unsigned (0).
            let value = if top { raw as i32 as f64 } else { raw as f64 };
            if double {
                cpu.vfp.set_f64(rd, value);
            } else {
                cpu.vfp.set_f32(rd, f64_to_f32_rmode(value, bits(cpu.vfp.fpscr, 22, 23), false));
            }
        }
        // VCVT to integer.
        (0b1100, _) | (0b1101, _) => {
            let source = if double {
                cpu.vfp.read_f64(rm)
            } else {
                cpu.vfp.read_f32(rm) as f64
            };
            let rounded = if top {
                source.trunc()
            } else {
                round_to_mode(source, bits(cpu.vfp.fpscr, 22, 23))
            };
            let signed = opc2 == 0b1101;
            let bits_out = if signed {
                saturate_to_i32(rounded) as u32
            } else {
                saturate_to_u32(rounded)
            };
            let sd = sreg(bits(op, 12, 15), bit(op, 22));
            cpu.vfp.regs[sd] = bits_out;
        }
        _ => {
            return Some(Exit::Undefined {
                pc: cpu.current_pc(),
                opcode: op,
            })
        }
    }
    None
}

/// VMOV (!second, !top), VABS (!second, top), VNEG (second,
/// !top) and VSQRT (second, top) on one register.
fn monadic(cpu: &mut Cpu, double: bool, second: bool, top: bool, rd: usize, rm: usize) {
    if double {
        // the moves and sign operations copy bits and flush nothing. titles
        // pass two singles in a double register, whose bits as a double can
        // be subnormal, and flushing them would zero the singles
        let raw = |cpu: &Cpu, index: usize| {
            (cpu.vfp.regs[(index * 2) & 31] as u64) | (cpu.vfp.regs[(index * 2 + 1) & 31] as u64) << 32
        };
        let bits = match (second, top) {
            (false, false) => raw(cpu, rm),
            (false, true) => raw(cpu, rm) & !(1 << 63),
            (true, false) => raw(cpu, rm) ^ (1 << 63),
            (true, true) => {
                let in_val = cpu.vfp.read_f64(rm);
                let v = sqrt_f64_rmode(in_val, bits(cpu.vfp.fpscr, 22, 23));
                if v.is_nan() && !in_val.is_nan() {
                    log::trace!(
                        target: "zakuro_cpu::vfp::nan",
                        "NaN from VSQRT of {} at 0x{:08X}, lr 0x{:08X}",
                        in_val,
                        cpu.current_pc(),
                        cpu.regs[14],
                    );
                }
                cpu.vfp.set_f64(rd, v);
                return;
            }
        };
        cpu.vfp.regs[(rd * 2) & 31] = bits as u32;
        cpu.vfp.regs[(rd * 2 + 1) & 31] = (bits >> 32) as u32;
    } else {
        // the sign operations work on the raw bits so NaN payloads survive.
        let raw = cpu.vfp.regs[rm];
        match (second, top) {
            (false, false) => cpu.vfp.regs[rd] = raw,
            (false, true) => cpu.vfp.regs[rd] = raw & 0x7FFF_FFFF,
            (true, false) => cpu.vfp.regs[rd] = raw ^ 0x8000_0000,
            (true, true) => {
                let in_val = cpu.vfp.read_f32(rm);
                let v = sqrt_f32_rmode(in_val, bits(cpu.vfp.fpscr, 22, 23));
                if v.is_nan() && !in_val.is_nan() {
                    log::trace!(
                        target: "zakuro_cpu::vfp::nan",
                        "NaN from VSQRT of {} at 0x{:08X}, lr 0x{:08X}",
                        in_val,
                        cpu.current_pc(),
                        cpu.regs[14],
                    );
                }
                cpu.vfp.set_f32(rd, v);
            }
        }
    }
}

fn compare(cpu: &mut Cpu, double: bool, rd: usize, rm: Option<usize>) {
    let (a, b) = if double {
        (
            cpu.vfp.read_f64(rd),
            rm.map_or(0.0, |r| cpu.vfp.read_f64(r)),
        )
    } else {
        (
            cpu.vfp.read_f32(rd) as f64,
            rm.map_or(0.0, |r| cpu.vfp.read_f32(r) as f64),
        )
    };

    // VFP comparison results map onto the same NZCV encoding the ARM
    // condition codes use, which is why VMRS APSR_nzcv works.
    let flags = if a.is_nan() || b.is_nan() {
        0b0011 // unordered, C and V set
    } else if a == b {
        0b0110 // z and C
    } else if a < b {
        0b1000 // n
    } else {
        0b0010 // c
    };

    cpu.vfp.fpscr = (cpu.vfp.fpscr & !FPSCR_FLAG_MASK) | (flags << 28);
}

fn adjust_f32_rmode(rn: f32, diff: f64, positive_exact: bool, rmode: u32) -> f32 {
    match rmode {
        0b01 => {
            if diff > 0.0 {
                rn.next_up()
            } else {
                rn
            }
        }
        0b10 => {
            if diff < 0.0 {
                rn.next_down()
            } else {
                rn
            }
        }
        0b11 => {
            if positive_exact && diff < 0.0 {
                rn.next_down()
            } else if !positive_exact && diff > 0.0 {
                rn.next_up()
            } else {
                rn
            }
        }
        _ => rn,
    }
}

fn add_f32_rmode(a: f32, b: f32, rmode: u32, fz: bool) -> f32 {
    if (rmode == 0 && !fz) || a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
        return a + b;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() && b.is_sign_negative() {
            -0.0
        } else if !a.is_sign_negative() && !b.is_sign_negative() {
            0.0
        } else if rmode == 0b10 {
            -0.0
        } else {
            0.0
        };
    }
    let (a64, b64) = (a as f64, b as f64);
    let s = a64 + b64;
    let bb = s - a64;
    let err = (a64 - (s - bb)) + (b64 - bb);
    if s == 0.0 && err == 0.0 {
        return if rmode == 0b10 { -0.0 } else { 0.0 };
    }
    let positive_exact = s > 0.0 || (s == 0.0 && err > 0.0);
    // Hardware FTZ / VFPv2 FZ detects tininess (< 2^-126) before rounding.
    let tiny = s.abs() < f32::MIN_POSITIVE as f64
        || (s.abs() == f32::MIN_POSITIVE as f64
            && ((s > 0.0 && err < 0.0) || (s < 0.0 && err > 0.0)));
    if fz && tiny {
        return if positive_exact { 0.0 } else { -0.0 };
    }
    let rn = s as f32;
    if rmode == 0 {
        return rn;
    }
    if rn.is_infinite() {
        return f64_to_f32_rmode(s, rmode, fz);
    }
    let diff = (s - rn as f64) + err;
    adjust_f32_rmode(rn, diff, positive_exact, rmode)
}

fn sub_f32_rmode(a: f32, b: f32, rmode: u32, fz: bool) -> f32 {
    if b.is_nan() {
        a - b
    } else {
        add_f32_rmode(a, f32::from_bits(b.to_bits() ^ 0x8000_0000), rmode, fz)
    }
}

fn mul_f32_rmode(a: f32, b: f32, rmode: u32, fz: bool) -> f32 {
    if (rmode == 0 && !fz)
        || a.is_nan()
        || b.is_nan()
        || a.is_infinite()
        || b.is_infinite()
        || a == 0.0
        || b == 0.0
    {
        a * b
    } else {
        f64_to_f32_rmode((a as f64) * (b as f64), rmode, fz)
    }
}

fn div_f32_rmode(a: f32, b: f32, rmode: u32, fz: bool) -> f32 {
    if (rmode == 0 && !fz)
        || a.is_nan()
        || b.is_nan()
        || a.is_infinite()
        || b.is_infinite()
        || a == 0.0
        || b == 0.0
    {
        return a / b;
    }
    let (a64, b64) = (a as f64, b as f64);
    if fz && a64.abs() < (f32::MIN_POSITIVE as f64) * b64.abs() {
        return if (a64 > 0.0) == (b64 > 0.0) { 0.0 } else { -0.0 };
    }
    if rmode == 0 {
        return a / b;
    }
    let q64 = a64 / b64;
    let rn = q64 as f32;
    if rn.is_infinite() {
        return f64_to_f32_rmode(q64, rmode, fz);
    }
    let rem = a64 - (rn as f64) * b64;
    let diff = if b64 > 0.0 { rem } else { -rem };
    adjust_f32_rmode(rn, diff, q64 > 0.0, rmode)
}

fn sqrt_f32_rmode(a: f32, rmode: u32) -> f32 {
    if rmode == 0 || a.is_nan() || a.is_infinite() || a <= 0.0 {
        return a.sqrt();
    }
    let a64 = a as f64;
    let rn = a64.sqrt() as f32;
    let diff = a64 - (rn as f64) * (rn as f64);
    adjust_f32_rmode(rn, diff, true, rmode)
}

fn f64_to_f32_rmode(d: f64, rmode: u32, fz: bool) -> f32 {
    let rn = d as f32;
    if d.is_nan() || d == 0.0 || d.is_infinite() {
        return rn;
    }
    if fz && d.abs() < f32::MIN_POSITIVE as f64 {
        return if d.is_sign_negative() { -0.0 } else { 0.0 };
    }
    if rmode == 0 {
        return rn;
    }
    if rn.is_infinite() {
        let towards_inf = match rmode {
            0b01 => d > 0.0,
            0b10 => d < 0.0,
            _ => false,
        };
        return if towards_inf {
            rn
        } else if d.is_sign_negative() {
            -f32::MAX
        } else {
            f32::MAX
        };
    }
    adjust_f32_rmode(rn, d - rn as f64, d > 0.0, rmode)
}

fn decode_f64(x: f64) -> (bool, i32, u64) {
    let bits = x.to_bits();
    let sign = (bits >> 63) != 0;
    let raw_exp = ((bits >> 52) & 0x7FF) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    if raw_exp == 0 {
        let shift = frac.leading_zeros() as i32 - 11;
        (sign, -1022 - shift, frac << (shift as u32))
    } else {
        (sign, raw_exp - 1023, (1u64 << 52) | frac)
    }
}

fn overflow_f64_rmode(sign: bool, rmode: u32) -> f64 {
    match rmode {
        0b00 => if sign { f64::NEG_INFINITY } else { f64::INFINITY },
        0b01 => if sign { -f64::MAX } else { f64::INFINITY },
        0b10 => if sign { f64::NEG_INFINITY } else { f64::MAX },
        _ => if sign { -f64::MAX } else { f64::MAX },
    }
}

fn pack_and_round_f64(
    sign: bool,
    mut exp: i32,
    sig: u128,
    top_bit: u32,
    sticky: bool,
    rmode: u32,
    fz: bool,
) -> f64 {
    if fz && exp < -1022 {
        return if sign { -0.0 } else { 0.0 };
    }
    let shift = if exp >= -1022 {
        top_bit as i32 - 52
    } else {
        top_bit as i32 - 52 + (-1022 - exp)
    };
    let (kept, round_bit, tail_sticky) = if shift <= 0 {
        ((sig << ((-shift) as u32)) as u64, false, sticky)
    } else if shift <= top_bit as i32 {
        let s = shift as u32;
        let kept = (sig >> s) as u64;
        let rb = ((sig >> (s - 1)) & 1) != 0;
        let mask = (1u128 << (s - 1)) - 1;
        let ts = sticky || (sig & mask) != 0;
        (kept, rb, ts)
    } else if shift == top_bit as i32 + 1 {
        let mask = (1u128 << top_bit) - 1;
        (0u64, true, sticky || (sig & mask) != 0)
    } else {
        (0u64, false, true)
    };

    let inexact = round_bit || tail_sticky;
    let inc = match rmode {
        0b00 => round_bit && (tail_sticky || (kept & 1) != 0),
        0b01 => !sign && inexact,
        0b10 => sign && inexact,
        _ => false,
    };
    let mut mant = kept + (inc as u64);
    let sign_bit = (sign as u64) << 63;
    if exp < -1022 {
        if mant >= (1u64 << 52) {
            return f64::from_bits(sign_bit | (1u64 << 52));
        }
        return f64::from_bits(sign_bit | mant);
    }
    if mant >= (1u64 << 53) {
        mant >>= 1;
        exp += 1;
    }
    if exp > 1023 {
        return overflow_f64_rmode(sign, rmode);
    }
    let raw_exp = ((exp + 1023) as u64) << 52;
    f64::from_bits(sign_bit | raw_exp | (mant & ((1u64 << 52) - 1)))
}

fn adjust_f64_rmode(rn: f64, diff_sign: i8, positive_exact: bool, rmode: u32) -> f64 {
    match rmode {
        0b01 => {
            if diff_sign > 0 {
                rn.next_up()
            } else {
                rn
            }
        }
        0b10 => {
            if diff_sign < 0 {
                rn.next_down()
            } else {
                rn
            }
        }
        0b11 => {
            if positive_exact && diff_sign < 0 {
                rn.next_down()
            } else if !positive_exact && diff_sign > 0 {
                rn.next_up()
            } else {
                rn
            }
        }
        _ => rn,
    }
}

fn add_f64_rmode(a: f64, b: f64, rmode: u32, fz: bool) -> f64 {
    if (rmode == 0 && !fz) || a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
        return a + b;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() && b.is_sign_negative() {
            -0.0
        } else if !a.is_sign_negative() && !b.is_sign_negative() {
            0.0
        } else if rmode == 0b10 {
            -0.0
        } else {
            0.0
        };
    }
    let scale = if (a + b).is_infinite() { 0.25 } else { 1.0 };
    let (a1, b1) = (a * scale, b * scale);
    let s1 = a1 + b1;
    let bb = s1 - a1;
    let err1 = (a1 - (s1 - bb)) + (b1 - bb);
    if s1 == 0.0 && err1 == 0.0 {
        return if rmode == 0b10 { -0.0 } else { 0.0 };
    }
    let positive_exact = s1 > 0.0 || (s1 == 0.0 && err1 > 0.0);
    if fz && scale == 1.0 {
        let tiny = s1.abs() < f64::MIN_POSITIVE
            || (s1.abs() == f64::MIN_POSITIVE
                && ((s1 > 0.0 && err1 < 0.0) || (s1 < 0.0 && err1 > 0.0)));
        if tiny {
            return if positive_exact { 0.0 } else { -0.0 };
        }
    }
    if rmode == 0 && scale == 1.0 {
        return s1;
    }
    let diff_sign = if err1 > 0.0 {
        1
    } else if err1 < 0.0 {
        -1
    } else {
        0
    };
    let rounded1 = adjust_f64_rmode(s1, diff_sign, positive_exact, rmode);
    if scale == 1.0 {
        rounded1
    } else {
        let unscaled = rounded1 * 4.0;
        if unscaled.is_infinite() {
            overflow_f64_rmode(!positive_exact, rmode)
        } else {
            unscaled
        }
    }
}

fn sub_f64_rmode(a: f64, b: f64, rmode: u32, fz: bool) -> f64 {
    if b.is_nan() {
        a - b
    } else {
        add_f64_rmode(a, f64::from_bits(b.to_bits() ^ (1u64 << 63)), rmode, fz)
    }
}

fn mul_f64_rmode(a: f64, b: f64, rmode: u32, fz: bool) -> f64 {
    if (rmode == 0 && !fz)
        || a.is_nan()
        || b.is_nan()
        || a.is_infinite()
        || b.is_infinite()
        || a == 0.0
        || b == 0.0
    {
        return a * b;
    }
    let (sa, ea, ma) = decode_f64(a);
    let (sb, eb, mb) = decode_f64(b);
    let prod = (ma as u128) * (mb as u128);
    let top = if (prod >> 105) != 0 { 105 } else { 104 };
    let exp = ea + eb + (top as i32 - 104);
    pack_and_round_f64(sa ^ sb, exp, prod, top, false, rmode, fz)
}

fn div_f64_rmode(a: f64, b: f64, rmode: u32, fz: bool) -> f64 {
    if (rmode == 0 && !fz)
        || a.is_nan()
        || b.is_nan()
        || a.is_infinite()
        || b.is_infinite()
        || a == 0.0
        || b == 0.0
    {
        return a / b;
    }
    let (sa, ea, ma) = decode_f64(a);
    let (sb, eb, mb) = decode_f64(b);
    let dividend = (ma as u128) << 55;
    let q = dividend / (mb as u128);
    let rem = dividend % (mb as u128);
    let top = if (q >> 55) != 0 { 55 } else { 54 };
    let exp = ea - eb + (top as i32 - 55);
    pack_and_round_f64(sa ^ sb, exp, q, top, rem != 0, rmode, fz)
}

fn sqrt_f64_rmode(a: f64, rmode: u32) -> f64 {
    if rmode == 0 || a.is_nan() || a.is_infinite() || a <= 0.0 {
        return a.sqrt();
    }
    let rn = a.sqrt();
    let (_, ea, ma) = decode_f64(a);
    let (_, ern, mrn) = decode_f64(rn);
    let rn_sq = (mrn as u128) * (mrn as u128);
    let diff_exp = (ea - 52) - (2 * ern - 104);
    let (lhs, rhs) = if diff_exp >= 0 {
        ((ma as u128) << (diff_exp as u32), rn_sq)
    } else {
        (ma as u128, rn_sq << ((-diff_exp) as u32))
    };
    let diff_sign = if lhs > rhs {
        1
    } else if lhs < rhs {
        -1
    } else {
        0
    };
    adjust_f64_rmode(rn, diff_sign, true, rmode)
}

fn round_to_mode(value: f64, mode: u32) -> f64 {
    match mode {
        0b00 => {
            // round to nearest, ties to even.
            let rounded = value.round();
            if (value - value.trunc()).abs() == 0.5 && rounded % 2.0 != 0.0 {
                rounded - value.signum()
            } else {
                rounded
            }
        }
        0b01 => value.ceil(),
        0b10 => value.floor(),
        _ => value.trunc(),
    }
}

fn saturate_to_i32(value: f64) -> i32 {
    if value.is_nan() {
        0
    } else if value >= i32::MAX as f64 {
        i32::MAX
    } else if value <= i32::MIN as f64 {
        i32::MIN
    } else {
        value as i32
    }
}

fn saturate_to_u32(value: f64) -> u32 {
    if value.is_nan() {
        0
    } else if value >= u32::MAX as f64 {
        u32::MAX
    } else if value <= 0.0 {
        0
    } else {
        value as u32
    }
}

// ---------------------------------------------------------------------------
// Register transfers (MCR / MRC on coprocessors 10 and 11)
// ---------------------------------------------------------------------------

pub fn coprocessor_register(cpu: &mut Cpu, op: u32) -> Option<Exit> {
    let load = bit(op, 20);
    let rt = bits(op, 12, 15) as usize;
    let opc1 = bits(op, 21, 23);

    // VMSR / VMRS, opc1 == 7, with CRn selecting the system register.
    if opc1 == 7 {
        let reg = bits(op, 16, 19);
        if load {
            let value = match reg {
                0 => 0x4100_0000, // FPSID, VFPv2, ARM implementer
                1 => cpu.vfp.fpscr,
                6 => 0,           // MVFR1
                7 => 0,           // MVFR0
                8 => cpu.vfp.fpexc,
                _ => 0,
            };
            if rt == 15 {
                // VMRS APSR_nzcv, FPSCR moves the comparison flags across.
                cpu.cpsr.n = bit(value, 31);
                cpu.cpsr.z = bit(value, 30);
                cpu.cpsr.c = bit(value, 29);
                cpu.cpsr.v = bit(value, 28);
            } else {
                cpu.regs[rt] = value;
            }
        } else {
            let value = cpu.regs[rt];
            match reg {
                1 => cpu.vfp.fpscr = value,
                8 => cpu.vfp.fpexc = value,
                _ => {}
            }
        }
        return None;
    }

    let coproc = bits(op, 8, 11);
    // VMOV between a core register and a single-precision register (`opc1 == 0`),
    // or upper half of a double-precision register (`FMRDH` / `FMDHR`, `coproc == 11 && opc1 == 1`).
    if (coproc == 10 || coproc == 11) && (opc1 == 0 || (coproc == 11 && opc1 == 1)) {
        let sn = sreg(bits(op, 16, 19), (coproc == 11 && opc1 == 1) || bit(op, 7));
        if load {
            let value = cpu.vfp.regs[sn];
            cpu.set_reg(rt, value);
        } else {
            cpu.vfp.regs[sn] = cpu.regs[rt];
        }
        return None;
    }

    log::debug!(
        "unimplemented VFP register transfer 0x{op:08X} at 0x{:08X}",
        cpu.current_pc()
    );
    None
}

// ---------------------------------------------------------------------------
// Load/store (LDC / STC space)
// ---------------------------------------------------------------------------

pub fn coprocessor_load_store<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    let coproc = bits(op, 8, 11);
    if coproc != 10 && coproc != 11 {
        log::warn!("LDC/STC for p{coproc} at 0x{:08X}", cpu.current_pc());
        return None;
    }

    // bits 27:21 == 0b1100010 is the 64-bit core<->VFP transfer space.
    if bits(op, 21, 27) == 0b110_0010 {
        return move_pair(cpu, op, coproc == 11);
    }

    let double = coproc == 11;
    let load = bit(op, 20);
    let pre_index = bit(op, 24);
    let add = bit(op, 23);
    let writeback = bit(op, 21);
    let rn = bits(op, 16, 19) as usize;
    let offset = (op & 0xFF) * 4;

    let d_bit = bit(op, 22);
    let vd = bits(op, 12, 15);
    let first = if double {
        dreg(vd, d_bit)
    } else {
        sreg(vd, d_bit)
    };

    // a single VLDR/VSTR has P=1, W=0, the multiple forms have P xor W.
    let count = if pre_index && !writeback {
        1
    } else if double {
        ((op & 0xFF) / 2) as usize
    } else {
        (op & 0xFF) as usize
    };

    let base = cpu.regs[rn];
    let start = if add {
        if pre_index && !writeback {
            base.wrapping_add(offset)
        } else {
            base
        }
    } else {
        base.wrapping_sub(offset)
    };

    let mut addr = start;
    for i in 0..count.max(1) {
        if double {
            let index = first + i;
            if load {
                let lo = bus.read32(addr);
                let hi = bus.read32(addr.wrapping_add(4));
                cpu.vfp.regs[(index * 2) & 31] = lo;
                cpu.vfp.regs[(index * 2 + 1) & 31] = hi;
            } else {
                bus.write32(addr, cpu.vfp.regs[(index * 2) & 31]);
                bus.write32(addr.wrapping_add(4), cpu.vfp.regs[(index * 2 + 1) & 31]);
            }
            addr = addr.wrapping_add(8);
        } else {
            let index = (first + i) & 31;
            if load {
                cpu.vfp.regs[index] = bus.read32(addr);
            } else {
                bus.write32(addr, cpu.vfp.regs[index]);
            }
            addr = addr.wrapping_add(4);
        }
    }

    if writeback {
        cpu.regs[rn] = if add {
            base.wrapping_add(offset)
        } else {
            base.wrapping_sub(offset)
        };
    }
    None
}

/// VMOV between two core registers and either a double or two singles.
fn move_pair(cpu: &mut Cpu, op: u32, double: bool) -> Option<Exit> {
    let load = bit(op, 20);
    let rt = bits(op, 12, 15) as usize;
    let rt2 = bits(op, 16, 19) as usize;
    let m_bit = bit(op, 5);
    let vm = op & 0xF;

    if double {
        let dm = dreg(vm, m_bit);
        if load {
            cpu.regs[rt] = cpu.vfp.regs[(dm * 2) & 31];
            cpu.regs[rt2] = cpu.vfp.regs[(dm * 2 + 1) & 31];
        } else {
            cpu.vfp.regs[(dm * 2) & 31] = cpu.regs[rt];
            cpu.vfp.regs[(dm * 2 + 1) & 31] = cpu.regs[rt2];
        }
    } else {
        let sm = sreg(vm, m_bit);
        if load {
            cpu.regs[rt] = cpu.vfp.regs[sm & 31];
            cpu.regs[rt2] = cpu.vfp.regs[(sm + 1) & 31];
        } else {
            cpu.vfp.regs[sm & 31] = cpu.regs[rt];
            cpu.vfp.regs[(sm + 1) & 31] = cpu.regs[rt2];
        }
    }
    None
}

/// kept so the default-NaN bit is not silently dropped when we add strict
/// NaN propagation.
pub const _FPSCR_DN: u32 = FPSCR_DN;
