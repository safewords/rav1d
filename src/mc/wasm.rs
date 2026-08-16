//! wasm32 SIMD128 kernels for the hot motion-compensation paths, 8 bpc.
//!
//! On native targets these paths dispatch to dav1d's hand-written SIMD; on
//! wasm there was only the scalar fallback, and MC is half of the decode
//! profile. These are the same arithmetic as the `_rust` functions in the
//! parent module — same widening, same rounding, same clipping — laid out
//! eight pixels per `v128`, so the output is bit-identical (the tests hold the
//! decoder to libdav1d's MD5s).
//!
//! Only 8-bit pixels and widths ≥ 8 come here; the callers in the parent
//! module fall through to the scalar code otherwise.

use core::arch::wasm32::*;
use std::mem::MaybeUninit;

use zerocopy::AsBytes;

use super::{get_filter, MID_STRIDE};
use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::headers::Rav1dFilterMode;
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;
use crate::strided::Strided as _;
use crate::tables::dav1d_mc_warp_filter;

/// 8-bit pixels: 8 · 2 taps of accumulator headroom before the rounding shift.
const INTERMEDIATE_BITS: u8 = 4;

/// The eight taps of a filter, each splatted across the i16 lanes.
struct Taps([v128; 8]);

impl Taps {
    #[inline(always)]
    fn new(f: &[i8; 8]) -> Self {
        Taps([
            i16x8_splat(f[0] as i16),
            i16x8_splat(f[1] as i16),
            i16x8_splat(f[2] as i16),
            i16x8_splat(f[3] as i16),
            i16x8_splat(f[4] as i16),
            i16x8_splat(f[5] as i16),
            i16x8_splat(f[6] as i16),
            i16x8_splat(f[7] as i16),
        ])
    }
}

/// Eight bytes at `p` widened to i16 lanes.
///
/// # Safety
///
/// `p..p+8` must be readable.
#[inline(always)]
unsafe fn load8_u8_as_i16(p: *const u8) -> v128 {
    // SAFETY: caller guarantees 8 readable bytes; wasm loads are unaligned.
    u16x8_extend_low_u8x16(unsafe { v128_load64_zero(p as *const u64) })
}

/// Eight i16 at `p`.
///
/// # Safety
///
/// `p..p+8` must be readable.
#[inline(always)]
unsafe fn load8_i16(p: *const i16) -> v128 {
    // SAFETY: caller guarantees 8 readable elements; wasm loads are unaligned.
    unsafe { v128_load(p as *const v128) }
}

/// Widening multiply-accumulate of one tap over eight lanes.
#[inline(always)]
fn mac(acc: (v128, v128), s: v128, tap: v128) -> (v128, v128) {
    (
        i32x4_add(acc.0, i32x4_extmul_low_i16x8(s, tap)),
        i32x4_add(acc.1, i32x4_extmul_high_i16x8(s, tap)),
    )
}

/// `(v + rnd) >> sh` on both halves, then packed to eight i16 lanes.
#[inline(always)]
fn round_pack_i16(acc: (v128, v128), rnd: i32, sh: u32) -> v128 {
    let r = i32x4_splat(rnd);
    i16x8_narrow_i32x4(
        i32x4_shr(i32x4_add(acc.0, r), sh),
        i32x4_shr(i32x4_add(acc.1, r), sh),
    )
}

/// Horizontal 8-tap over `src[x..x+15]` for eight output pixels — raw sums.
///
/// # Safety
///
/// `src..src+15` must be readable.
#[inline(always)]
unsafe fn h8(src: *const u8, taps: &Taps) -> (v128, v128) {
    let mut acc = (i32x4_splat(0), i32x4_splat(0));
    for k in 0..8 {
        // SAFETY: k ≤ 7, so src+k..src+k+8 ⊂ src..src+15.
        let s = unsafe { load8_u8_as_i16(src.add(k)) };
        acc = mac(acc, s, taps.0[k]);
    }
    acc
}

/// Vertical 8-tap over eight u8 rows (`rows[k]` is row `y+k-3`) — raw sums.
///
/// # Safety
///
/// Each `rows[k]..+8` must be readable.
#[inline(always)]
unsafe fn v8_u8(rows: &[*const u8; 8], taps: &Taps) -> (v128, v128) {
    let mut acc = (i32x4_splat(0), i32x4_splat(0));
    for k in 0..8 {
        // SAFETY: caller guarantees each row pointer has 8 readable bytes.
        let s = unsafe { load8_u8_as_i16(rows[k]) };
        acc = mac(acc, s, taps.0[k]);
    }
    acc
}

/// The horizontal-pass intermediate for the 2-D filters: `(h + 7)` rows of
/// `w` i16 at a stride of [`MID_STRIDE`]. Left uninitialised — the scalar
/// path zeroes 34 KB per call, which for an 8×8 block is more work than the
/// filter — and only ever read where the horizontal pass has written.
struct Mid(MaybeUninit<[[i16; MID_STRIDE]; 135]>);

impl Mid {
    #[inline(always)]
    fn new() -> Self {
        Mid(MaybeUninit::uninit())
    }

    /// Store eight i16 at row `y`, column `x`.
    #[inline(always)]
    fn store8(&mut self, y: usize, x: usize, v: v128) {
        debug_assert!(y < 135 && x + 8 <= MID_STRIDE);
        // SAFETY: (y, x..x+8) is inside the array; unaligned store.
        unsafe {
            v128_store(
                (self.0.as_mut_ptr() as *mut i16).add(y * MID_STRIDE + x) as *mut v128,
                v,
            )
        }
    }

    /// Vertical 8-tap over rows `y..y+8` at column `x` — raw sums. The caller
    /// guarantees those rows/columns were stored by the horizontal pass.
    #[inline(always)]
    fn v8(&self, y: usize, x: usize, taps: &Taps) -> (v128, v128) {
        debug_assert!(y + 8 <= 135 && x + 8 <= MID_STRIDE);
        let mut acc = (i32x4_splat(0), i32x4_splat(0));
        for k in 0..8 {
            // SAFETY: in bounds by the assert; the horizontal pass wrote every
            // (row < h+7, col < w) cell before any vertical read (y+k < h+7, x+8 ≤ w).
            let s =
                unsafe { load8_i16((self.0.as_ptr() as *const i16).add((y + k) * MID_STRIDE + x)) };
            acc = mac(acc, s, taps.0[k]);
        }
        acc
    }
}

/// Store eight i16 lanes as eight clipped u8 pixels.
///
/// # Safety
///
/// `dst..dst+8` must be writable.
#[inline(always)]
unsafe fn store8_u8(dst: *mut u8, v: v128) {
    let p = u8x16_narrow_i16x8(v, v);
    // SAFETY: caller guarantees 8 writable bytes; wasm stores are unaligned.
    unsafe { v128_store64_lane::<0>(p, dst as *mut u64) }
}

/// Store eight i16 lanes.
///
/// # Safety
///
/// `dst..dst+8` must be writable.
#[inline(always)]
unsafe fn store8_i16(dst: *mut i16, v: v128) {
    // SAFETY: caller guarantees 8 writable elements; wasm stores are unaligned.
    unsafe { v128_store(dst as *mut v128, v) }
}

/// The 8-bpc, `w >= 8` half of `put_8tap_rust`. Returns `false` (having done
/// nothing) for the copy case, which the caller handles.
#[inline(never)]
pub(super) fn put_8tap<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    src: Rav1dPictureDataComponentOffset,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    (h_filter_type, v_filter_type): (Rav1dFilterMode, Rav1dFilterMode),
) -> bool {
    debug_assert!(BD::BITDEPTH == 8 && w >= 8 && w % 8 == 0);
    let intermediate_rnd = 32 + ((1 << (6 - INTERMEDIATE_BITS)) >> 1);
    let src_stride = src.pixel_stride::<BD>();
    let dst_stride = dst.pixel_stride::<BD>();

    let fh = get_filter(mx, w, h_filter_type);
    let fv = get_filter(my, h, v_filter_type);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let th = Taps::new(fh);
            let tv = Taps::new(fv);
            let tmp_h = h + 7;
            let mut mid = Mid::new();
            for y in 0..tmp_h {
                let row = src + (y as isize - 3) * src_stride - 3usize;
                let row = &*row.slice::<BD>(w + 7);
                let row = row.as_bytes();
                for x in (0..w).step_by(8) {
                    // SAFETY: row has w+7 bytes; x+15 ≤ w+7 since x+8 ≤ w.
                    let acc = unsafe { h8(row.as_ptr().add(x), &th) };
                    let v = round_pack_i16(
                        acc,
                        (1 << (6 - INTERMEDIATE_BITS)) >> 1,
                        (6 - INTERMEDIATE_BITS) as u32,
                    );
                    mid.store8(y, x, v);
                }
            }
            for y in 0..h {
                let drow = dst + y as isize * dst_stride;
                let drow = &mut *drow.slice_mut::<BD>(w);
                let drow = drow.as_bytes_mut();
                for x in (0..w).step_by(8) {
                    let acc = mid.v8(y, x, &tv);
                    let v = round_pack_i16(
                        acc,
                        (1 << (6 + INTERMEDIATE_BITS)) >> 1,
                        (6 + INTERMEDIATE_BITS) as u32,
                    );
                    // SAFETY: drow has w bytes and x+8 ≤ w.
                    unsafe { store8_u8(drow.as_mut_ptr().add(x), v) };
                }
            }
        }
        (Some(fh), None) => {
            let th = Taps::new(fh);
            for y in 0..h {
                let row = src + y as isize * src_stride - 3usize;
                let row = &*row.slice::<BD>(w + 7);
                let row = row.as_bytes();
                let drow = dst + y as isize * dst_stride;
                let drow = &mut *drow.slice_mut::<BD>(w);
                let drow = drow.as_bytes_mut();
                for x in (0..w).step_by(8) {
                    // SAFETY: as above.
                    let acc = unsafe { h8(row.as_ptr().add(x), &th) };
                    let v = round_pack_i16(acc, intermediate_rnd, 6);
                    // SAFETY: as above.
                    unsafe { store8_u8(drow.as_mut_ptr().add(x), v) };
                }
            }
        }
        (None, Some(fv)) => {
            let tv = Taps::new(fv);
            for y in 0..h {
                let drow = dst + y as isize * dst_stride;
                let drow = &mut *drow.slice_mut::<BD>(w);
                let drow = drow.as_bytes_mut();
                // Hold all eight source rows as byte slices for the whole row.
                let rows: [_; 8] = std::array::from_fn(|k| {
                    let r = src + (y as isize + k as isize - 3) * src_stride;
                    r.slice::<BD>(w)
                });
                let ptrs: [*const u8; 8] = std::array::from_fn(|k| rows[k].as_bytes().as_ptr());
                for x in (0..w).step_by(8) {
                    let at: [*const u8; 8] = std::array::from_fn(|k| {
                        // SAFETY: each row has w bytes and x+8 ≤ w.
                        unsafe { ptrs[k].add(x) }
                    });
                    // SAFETY: as above.
                    let acc = unsafe { v8_u8(&at, &tv) };
                    let v = round_pack_i16(acc, (1 << 6) >> 1, 6);
                    // SAFETY: as above.
                    unsafe { store8_u8(drow.as_mut_ptr().add(x), v) };
                }
            }
        }
        (None, None) => return false,
    }
    true
}

/// The 8-bpc, `w >= 8` half of `prep_8tap_rust`. Same contract as [`put_8tap`].
#[inline(never)]
pub(super) fn prep_8tap<BD: BitDepth>(
    tmp: &mut [i16],
    src: Rav1dPictureDataComponentOffset,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    (h_filter_type, v_filter_type): (Rav1dFilterMode, Rav1dFilterMode),
) -> bool {
    debug_assert!(BD::BITDEPTH == 8 && w >= 8 && w % 8 == 0);
    let src_stride = src.pixel_stride::<BD>();
    let fh = get_filter(mx, w, h_filter_type);
    let fv = get_filter(my, h, v_filter_type);
    let h_rnd = (1 << (6 - INTERMEDIATE_BITS)) >> 1;
    let h_sh = (6 - INTERMEDIATE_BITS) as u32;

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let th = Taps::new(fh);
            let tv = Taps::new(fv);
            let tmp_h = h + 7;
            let mut mid = Mid::new();
            for y in 0..tmp_h {
                let row = src + (y as isize - 3) * src_stride - 3usize;
                let row = &*row.slice::<BD>(w + 7);
                let row = row.as_bytes();
                for x in (0..w).step_by(8) {
                    // SAFETY: row has w+7 bytes; x+15 ≤ w+7.
                    let acc = unsafe { h8(row.as_ptr().add(x), &th) };
                    let v = round_pack_i16(acc, h_rnd, h_sh);
                    mid.store8(y, x, v);
                }
            }
            for y in 0..h {
                let trow = &mut tmp[y * w..][..w];
                for x in (0..w).step_by(8) {
                    let acc = mid.v8(y, x, &tv);
                    // PREP_BIAS is 0 at 8 bpc, so `.sub_prep_bias()` is the identity.
                    let v = round_pack_i16(acc, (1 << 6) >> 1, 6);
                    // SAFETY: trow has w elements and x+8 ≤ w.
                    unsafe { store8_i16(trow.as_mut_ptr().add(x), v) };
                }
            }
        }
        (Some(fh), None) => {
            let th = Taps::new(fh);
            for y in 0..h {
                let row = src + y as isize * src_stride - 3usize;
                let row = &*row.slice::<BD>(w + 7);
                let row = row.as_bytes();
                let trow = &mut tmp[y * w..][..w];
                for x in (0..w).step_by(8) {
                    // SAFETY: as above.
                    let acc = unsafe { h8(row.as_ptr().add(x), &th) };
                    let v = round_pack_i16(acc, h_rnd, h_sh);
                    // SAFETY: as above.
                    unsafe { store8_i16(trow.as_mut_ptr().add(x), v) };
                }
            }
        }
        (None, Some(fv)) => {
            let tv = Taps::new(fv);
            for y in 0..h {
                let trow = &mut tmp[y * w..][..w];
                let rows: [_; 8] = std::array::from_fn(|k| {
                    let r = src + (y as isize + k as isize - 3) * src_stride;
                    r.slice::<BD>(w)
                });
                let ptrs: [*const u8; 8] = std::array::from_fn(|k| rows[k].as_bytes().as_ptr());
                for x in (0..w).step_by(8) {
                    let at: [*const u8; 8] = std::array::from_fn(|k| {
                        // SAFETY: each row has w bytes and x+8 ≤ w.
                        unsafe { ptrs[k].add(x) }
                    });
                    // SAFETY: as above.
                    let acc = unsafe { v8_u8(&at, &tv) };
                    let v = round_pack_i16(acc, h_rnd, h_sh);
                    // SAFETY: as above.
                    unsafe { store8_i16(trow.as_mut_ptr().add(x), v) };
                }
            }
        }
        (None, None) => {
            // `prep_rust`: tmp = src << intermediate_bits.
            for y in 0..h {
                let row = src + y as isize * src_stride;
                let row = &*row.slice::<BD>(w);
                let row = row.as_bytes();
                let trow = &mut tmp[y * w..][..w];
                for x in (0..w).step_by(8) {
                    // SAFETY: row has w bytes and x+8 ≤ w.
                    let s = unsafe { load8_u8_as_i16(row.as_ptr().add(x)) };
                    let v = i16x8_shl(s, INTERMEDIATE_BITS as u32);
                    // SAFETY: trow has w elements and x+8 ≤ w.
                    unsafe { store8_i16(trow.as_mut_ptr().add(x), v) };
                }
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Warped motion, 8x8
// ---------------------------------------------------------------------------

/// Transpose eight rows of eight i16 lanes.
#[inline(always)]
fn transpose8x8_i16(r: [v128; 8]) -> [v128; 8] {
    // 16-bit interleave.
    let a0 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(r[0], r[1]);
    let a1 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(r[0], r[1]);
    let a2 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(r[2], r[3]);
    let a3 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(r[2], r[3]);
    let a4 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(r[4], r[5]);
    let a5 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(r[4], r[5]);
    let a6 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(r[6], r[7]);
    let a7 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(r[6], r[7]);
    // 32-bit interleave.
    let b0 = i32x4_shuffle::<0, 4, 1, 5>(a0, a2);
    let b1 = i32x4_shuffle::<2, 6, 3, 7>(a0, a2);
    let b2 = i32x4_shuffle::<0, 4, 1, 5>(a1, a3);
    let b3 = i32x4_shuffle::<2, 6, 3, 7>(a1, a3);
    let b4 = i32x4_shuffle::<0, 4, 1, 5>(a4, a6);
    let b5 = i32x4_shuffle::<2, 6, 3, 7>(a4, a6);
    let b6 = i32x4_shuffle::<0, 4, 1, 5>(a5, a7);
    let b7 = i32x4_shuffle::<2, 6, 3, 7>(a5, a7);
    // 64-bit interleave.
    [
        i64x2_shuffle::<0, 2>(b0, b4),
        i64x2_shuffle::<1, 3>(b0, b4),
        i64x2_shuffle::<0, 2>(b1, b5),
        i64x2_shuffle::<1, 3>(b1, b5),
        i64x2_shuffle::<0, 2>(b2, b6),
        i64x2_shuffle::<1, 3>(b2, b6),
        i64x2_shuffle::<0, 2>(b3, b7),
        i64x2_shuffle::<1, 3>(b3, b7),
    ]
}

/// The eight taps of `dav1d_mc_warp_filter[idx]` as i16 lanes.
#[inline(always)]
fn warp_taps(idx: i32) -> v128 {
    let f = &dav1d_mc_warp_filter[idx as usize];
    // SAFETY: `f` is exactly 8 bytes.
    i16x8_extend_low_i8x16(unsafe { v128_load64_zero(f.as_ptr() as *const u64) })
}

/// Sum the eight lanes of `dot(a, f)` for four (a, f) pairs → one i32x4.
#[inline(always)]
fn dot4(a: [v128; 4], f: [v128; 4]) -> v128 {
    let d0 = i32x4_dot_i16x8(a[0], f[0]);
    let d1 = i32x4_dot_i16x8(a[1], f[1]);
    let d2 = i32x4_dot_i16x8(a[2], f[2]);
    let d3 = i32x4_dot_i16x8(a[3], f[3]);
    // [d0.0+d0.2, d0.1+d0.3, d1.0+d1.2, d1.1+d1.3], same for d2/d3.
    let t0 = i32x4_add(
        i32x4_shuffle::<0, 1, 4, 5>(d0, d1),
        i32x4_shuffle::<2, 3, 6, 7>(d0, d1),
    );
    let t1 = i32x4_add(
        i32x4_shuffle::<0, 1, 4, 5>(d2, d3),
        i32x4_shuffle::<2, 3, 6, 7>(d2, d3),
    );
    i32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(t0, t1),
        i32x4_shuffle::<1, 3, 5, 7>(t0, t1),
    )
}

/// The horizontal pass shared by `warp_affine_8x8` and `warp_affine_8x8t`:
/// fifteen rows of eight, each pixel with its own filter, into a transposed
/// intermediate (`mid_t[x][y]`, 16 rows so column windows can be loaded).
#[inline(always)]
fn warp_h_pass<BD: BitDepth>(
    src: Rav1dPictureDataComponentOffset,
    abcd: &[i16; 4],
    mx: i32,
) -> [[i16; 16]; 8] {
    let stride = src.pixel_stride::<BD>();
    let rnd = (1 << (7 - INTERMEDIATE_BITS)) >> 1;
    let sh = (7 - INTERMEDIATE_BITS) as u32;
    let mut mid = [i16x8_splat(0); 16];
    for y in 0..15 {
        let row = src + (y as isize - 3) * stride - 3usize;
        let row = &*row.slice::<BD>(15);
        let row = row.as_bytes();
        let mx = mx + y as i32 * abcd[1] as i32;
        let taps = |x: usize| warp_taps(64 + ((mx + x as i32 * abcd[0] as i32 + 512) >> 10));
        // SAFETY: row has 15 bytes; x+8 ≤ 15 for x ≤ 7.
        let px = |x: usize| unsafe { load8_u8_as_i16(row.as_ptr().add(x)) };
        let lo = dot4(
            [px(0), px(1), px(2), px(3)],
            [taps(0), taps(1), taps(2), taps(3)],
        );
        let hi = dot4(
            [px(4), px(5), px(6), px(7)],
            [taps(4), taps(5), taps(6), taps(7)],
        );
        mid[y] = round_pack_i16((lo, hi), rnd, sh);
    }
    let t0 = transpose8x8_i16([
        mid[0], mid[1], mid[2], mid[3], mid[4], mid[5], mid[6], mid[7],
    ]);
    let t1 = transpose8x8_i16([
        mid[8], mid[9], mid[10], mid[11], mid[12], mid[13], mid[14], mid[15],
    ]);
    let mut out = [[0i16; 16]; 8];
    for x in 0..8 {
        // SAFETY: each destination is 16 i16 = 32 bytes.
        unsafe {
            store8_i16(out[x].as_mut_ptr(), t0[x]);
            store8_i16(out[x].as_mut_ptr().add(8), t1[x]);
        }
    }
    out
}

/// One output row of the vertical warp pass: raw sums for eight pixels.
#[inline(always)]
fn warp_v_row(mid_t: &[[i16; 16]; 8], y: usize, abcd: &[i16; 4], my: i32) -> (v128, v128) {
    let my = my + y as i32 * abcd[3] as i32;
    let taps = |x: usize| warp_taps(64 + ((my + x as i32 * abcd[2] as i32 + 512) >> 10));
    // SAFETY: mid_t[x] has 16 elements; y+8 ≤ 15 for y ≤ 7.
    let col = |x: usize| unsafe { load8_i16(mid_t[x].as_ptr().add(y)) };
    (
        dot4(
            [col(0), col(1), col(2), col(3)],
            [taps(0), taps(1), taps(2), taps(3)],
        ),
        dot4(
            [col(4), col(5), col(6), col(7)],
            [taps(4), taps(5), taps(6), taps(7)],
        ),
    )
}

/// 8-bpc `warp_affine_8x8_rust`.
#[inline(never)]
pub(super) fn warp_affine_8x8<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    src: Rav1dPictureDataComponentOffset,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    debug_assert!(BD::BITDEPTH == 8);
    let mid_t = warp_h_pass::<BD>(src, abcd, mx);
    let dst_stride = dst.pixel_stride::<BD>();
    let rnd = (1 << (7 + INTERMEDIATE_BITS)) >> 1;
    let sh = (7 + INTERMEDIATE_BITS) as u32;
    for y in 0..8 {
        let acc = warp_v_row(&mid_t, y, abcd, my);
        let v = round_pack_i16(acc, rnd, sh);
        let drow = dst + y as isize * dst_stride;
        let drow = &mut *drow.slice_mut::<BD>(8);
        let drow = drow.as_bytes_mut();
        // SAFETY: drow is 8 bytes.
        unsafe { store8_u8(drow.as_mut_ptr(), v) };
    }
}

/// 8-bpc `warp_affine_8x8t_rust`.
#[inline(never)]
pub(super) fn warp_affine_8x8t<BD: BitDepth>(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: Rav1dPictureDataComponentOffset,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    debug_assert!(BD::BITDEPTH == 8);
    let mid_t = warp_h_pass::<BD>(src, abcd, mx);
    for y in 0..8 {
        let acc = warp_v_row(&mid_t, y, abcd, my);
        // PREP_BIAS is 0 at 8 bpc.
        let v = round_pack_i16(acc, (1 << 7) >> 1, 7);
        let trow = &mut tmp[y * tmp_stride..][..8];
        // SAFETY: trow is 8 elements.
        unsafe { store8_i16(trow.as_mut_ptr(), v) };
    }
}
