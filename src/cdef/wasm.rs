//! wasm32 SIMD128 kernel for the CDEF filter block, 8 bpc.
//!
//! Same arithmetic as `cdef_filter_block_rust` — the padded i16 `tmp` block,
//! `constrain()`, the pri/sec taps, the min/max clamp — one row of eight (or
//! four) pixels per `v128`. Bit-identical to the scalar path.
//!
//! The one subtlety is the padding marker `i16::MIN` (`fill()`): the scalar
//! code treats it as "very large" by comparing as unsigned for `min` and as
//! signed for `max`, and its `constrain()` runs in i32 where `|MIN - px|`
//! is simply huge and contributes 0. In i16 lanes `abs(MIN)` is `MIN` again,
//! so the absolute difference is handled as *unsigned* (`u16x8_shr`,
//! `u16x8_min`), which makes 0x8000 the huge value it is meant to be and
//! sends the tap's contribution to 0 exactly as the scalar code does.

use core::arch::wasm32::*;
use std::ffi::c_int;

use zerocopy::AsBytes;

use super::TMP_STRIDE;
use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;
use crate::strided::Strided as _;
use crate::tables::DAV1D_CDEF_DIRECTIONS;

/// `constrain(p - px, strength, shift)` over eight lanes, plus the running
/// unsigned-min / signed-max the clamp needs (only used when both strengths
/// are on).
#[inline(always)]
fn constrain8(p: v128, px: v128, strength: v128, shift: u32) -> v128 {
    let diff = i16x8_sub_sat(p, px);
    let adiff = i16x8_abs(diff); // 0x8000 stays 0x8000: read it as unsigned below.
    let t = i16x8_max(i16x8_splat(0), i16x8_sub(strength, u16x8_shr(adiff, shift)));
    let m = u16x8_min(adiff, t);
    // apply_sign(m, diff): (m ^ s) - s with s = diff >> 15.
    let s = i16x8_shr(diff, 15);
    i16x8_sub(v128_xor(m, s), s)
}

/// Load eight i16 of the padded block at `base + off`.
#[inline(always)]
fn tap(tmp: &[i16], base: usize, off: isize) -> v128 {
    let i = base.wrapping_add_signed(off);
    let s = &tmp[i..i + 8];
    // SAFETY: `s` is exactly 8 i16; wasm loads are unaligned.
    unsafe { v128_load(s.as_ptr() as *const v128) }
}

/// 8-bpc `cdef_filter_block_rust` after `padding()` has filled `tmp`.
#[inline(never)]
pub(super) fn cdef_filter_block<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    tmp: &[i16; TMP_STRIDE * TMP_STRIDE],
    pri_strength: c_int,
    sec_strength: c_int,
    dir: usize,
    damping: c_int,
    w: usize,
    h: usize,
) {
    debug_assert!(BD::BITDEPTH == 8 && (w == 4 || w == 8));
    let stride = dst.pixel_stride::<BD>();
    let tmp_offset = 2 * TMP_STRIDE + 2;

    let pri_tap = 4 - (pri_strength & 1);
    let pri_shift = if pri_strength != 0 {
        std::cmp::max(0, damping - pri_strength.ilog2() as c_int) as u32
    } else {
        0
    };
    let sec_shift = if sec_strength != 0 {
        (damping - sec_strength.ilog2() as c_int) as u32
    } else {
        0
    };
    let v_pri = i16x8_splat(pri_strength as i16);
    let v_sec = i16x8_splat(sec_strength as i16);
    let pri_taps = [
        i16x8_splat(pri_tap as i16),
        i16x8_splat((pri_tap & 3 | 2) as i16),
    ];
    let sec_taps = [i16x8_splat(2), i16x8_splat(1)];
    let zero = i16x8_splat(0);
    let eight = i16x8_splat(8);

    for y in 0..h {
        let base = y * TMP_STRIDE + tmp_offset;
        let drow = dst + y as isize * stride;
        let drow = &mut *drow.slice_mut::<BD>(w);
        let drow = drow.as_bytes_mut();
        // SAFETY: drow has w ≥ 4 bytes; a 4-byte load covers w == 4 and the
        // upper lanes are never stored.
        let px = if w == 8 {
            // SAFETY: 8 bytes.
            u16x8_extend_low_u8x16(unsafe { v128_load64_zero(drow.as_ptr() as *const u64) })
        } else {
            // SAFETY: 4 bytes.
            u16x8_extend_low_u8x16(unsafe { v128_load32_zero(drow.as_ptr() as *const u32) })
        };

        let mut sum = zero;
        let mut min = px;
        let mut max = px;
        let both = pri_strength != 0 && sec_strength != 0;
        for k in 0..2 {
            if pri_strength != 0 {
                let off1 = DAV1D_CDEF_DIRECTIONS[dir + 2][k] as isize;
                let p0 = tap(tmp, base, off1);
                let p1 = tap(tmp, base, -off1);
                let c = i16x8_add(
                    constrain8(p0, px, v_pri, pri_shift),
                    constrain8(p1, px, v_pri, pri_shift),
                );
                sum = i16x8_add(sum, i16x8_mul(pri_taps[k], c));
                if both {
                    min = u16x8_min(min, u16x8_min(p0, p1));
                    max = i16x8_max(max, i16x8_max(p0, p1));
                }
            }
            if sec_strength != 0 {
                let off2 = DAV1D_CDEF_DIRECTIONS[dir + 4][k] as isize;
                let off3 = DAV1D_CDEF_DIRECTIONS[dir][k] as isize;
                let s0 = tap(tmp, base, off2);
                let s1 = tap(tmp, base, -off2);
                let s2 = tap(tmp, base, off3);
                let s3 = tap(tmp, base, -off3);
                let c = i16x8_add(
                    i16x8_add(
                        constrain8(s0, px, v_sec, sec_shift),
                        constrain8(s1, px, v_sec, sec_shift),
                    ),
                    i16x8_add(
                        constrain8(s2, px, v_sec, sec_shift),
                        constrain8(s3, px, v_sec, sec_shift),
                    ),
                );
                sum = i16x8_add(sum, i16x8_mul(sec_taps[k], c));
                if both {
                    min = u16x8_min(min, u16x8_min(u16x8_min(s0, s1), u16x8_min(s2, s3)));
                    max = i16x8_max(max, i16x8_max(i16x8_max(s0, s1), i16x8_max(s2, s3)));
                }
            }
        }
        // px + ((sum - (sum < 0) + 8) >> 4)
        let neg = i16x8_shr(sum, 15); // -1 where sum < 0
        let adj = i16x8_shr(i16x8_add(i16x8_add(sum, neg), eight), 4);
        let mut out = i16x8_add(px, adj);
        if both {
            out = i16x8_min(i16x8_max(out, min), max);
        }
        let packed = u8x16_narrow_i16x8(out, out);
        if w == 8 {
            // SAFETY: drow has 8 bytes.
            unsafe { v128_store64_lane::<0>(packed, drow.as_mut_ptr() as *mut u64) };
        } else {
            // SAFETY: drow has 4 bytes.
            unsafe { v128_store32_lane::<0>(packed, drow.as_mut_ptr() as *mut u32) };
        }
    }
}
