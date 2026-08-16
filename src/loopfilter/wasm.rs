//! wasm32 SIMD128 kernel for the deblocking loop filter, 8 bpc.
//!
//! `loop_filter` filters four positions along one edge with one set of
//! thresholds; here those four positions are the low four i16 lanes of a
//! `v128` and the scalar `if` chain becomes per-lane masks and `bitselect`s.
//! Same arithmetic, same clipping, so bit-identical to the scalar path.
//!
//! Two orientations, decided by the strides: taps in rows and positions along
//! a row (4-byte loads/stores per tap), or taps along a row and positions in
//! rows (a small gather/scatter per row).

use core::arch::wasm32::*;
use std::ffi::c_int;

use zerocopy::AsBytes;

use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;

/// The fourteen taps p6..p0, q0..q6 as i16 lanes (lanes 0..4 valid), indexed
/// 0..14 (p6 = 0, p0 = 6, q0 = 7, q6 = 13). Only `half` on each side is loaded.
struct Taps {
    v: [v128; 14],
}

const P: usize = 6; // index of p0
const Q: usize = 7; // index of q0

impl Taps {
    #[inline(always)]
    fn p(&self, k: usize) -> v128 {
        self.v[P - k]
    }
    #[inline(always)]
    fn q(&self, k: usize) -> v128 {
        self.v[Q + k]
    }
}

/// Positions along a row (`stridea == 1`), taps down the rows (`strideb == stride`).
#[inline(always)]
fn load_rows<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    strideb: isize,
    half: usize,
) -> Taps {
    let mut v = [i16x8_splat(0); 14];
    for k in 0..2 * half {
        let off = k as isize - half as isize; // -half..half-1  ⇒ p(half-1)..q(half-1)
        let row = dst + off * strideb;
        let row = &*row.slice::<BD>(4);
        let row = row.as_bytes();
        // SAFETY: 4 bytes.
        v[P - half + 1 + k] =
            u16x8_extend_low_u8x16(unsafe { v128_load32_zero(row.as_ptr() as *const u32) });
    }
    Taps { v }
}

/// Positions down the rows (`stridea == stride`), taps along a row (`strideb == 1`).
#[inline(always)]
fn load_cols<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    stridea: isize,
    half: usize,
) -> Taps {
    let mut bytes = [[0u8; 14]; 4];
    for j in 0..4 {
        let row = dst + j as isize * stridea - half;
        let row = &*row.slice::<BD>(2 * half);
        bytes[j][..2 * half].copy_from_slice(row.as_bytes());
    }
    let mut v = [i16x8_splat(0); 14];
    for k in 0..2 * half {
        v[P - half + 1 + k] = i16x8(
            bytes[0][k] as i16,
            bytes[1][k] as i16,
            bytes[2][k] as i16,
            bytes[3][k] as i16,
            0,
            0,
            0,
            0,
        );
    }
    Taps { v }
}

#[inline(always)]
fn absdiff(a: v128, b: v128) -> v128 {
    i16x8_abs(i16x8_sub(a, b))
}

/// `|a - b| <= t` per lane.
#[inline(always)]
fn le_thresh(a: v128, b: v128, t: v128) -> v128 {
    i16x8_le(absdiff(a, b), t)
}

#[inline(always)]
fn shr_rnd(sum: v128, add: i16, sh: u32) -> v128 {
    i16x8_shr(i16x8_add(sum, i16x8_splat(add)), sh)
}

/// Sum of a list of taps (with repeats), i16 lanes.
#[inline(always)]
fn sum(terms: &[v128]) -> v128 {
    let mut acc = terms[0];
    for t in &terms[1..] {
        acc = i16x8_add(acc, *t);
    }
    acc
}

/// 8-bpc `loop_filter`: `wd` in {4, 6, 8, 16}, thresholds at 8-bit scale.
#[inline(never)]
pub(super) fn loop_filter<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    e: u8,
    i: u8,
    h: u8,
    stridea: isize,
    strideb: isize,
    wd: c_int,
) {
    debug_assert!(BD::BITDEPTH == 8);
    let half = (wd / 2) as usize; // taps needed on each side: 2, 3, 4, 8 (wd 16 uses 7)
    let half = half.min(7);
    let by_rows = stridea == 1;
    let t = if by_rows {
        load_rows::<BD>(dst, strideb, half)
    } else {
        load_cols::<BD>(dst, stridea, half)
    };
    let (p1, p0, q0, q1) = (t.p(1), t.p(0), t.q(0), t.q(1));

    let ve = i16x8_splat(e as i16);
    let vi = i16x8_splat(i as i16);
    let vh = i16x8_splat(h as i16);
    let vf = i16x8_splat(1); // flatness threshold `f` at 8 bpc

    // fm: the "filter mask".
    let mut fm = v128_and(
        v128_and(le_thresh(p1, p0, vi), le_thresh(q1, q0, vi)),
        i16x8_le(
            i16x8_add(i16x8_shl(absdiff(p0, q0), 1), i16x8_shr(absdiff(p1, q1), 1)),
            ve,
        ),
    );
    if wd > 4 {
        let (p2, q2) = (t.p(2), t.q(2));
        fm = v128_and(fm, v128_and(le_thresh(p2, p1, vi), le_thresh(q2, q1, vi)));
        if wd > 6 {
            let (p3, q3) = (t.p(3), t.q(3));
            fm = v128_and(fm, v128_and(le_thresh(p3, p2, vi), le_thresh(q3, q2, vi)));
        }
    }
    // Only the four low lanes are positions; ignore the rest of the mask.
    if !v128_any_true(v128_and(fm, i64x2(-1, 0))) {
        return;
    }

    let mut out = t.v; // updated taps, then written back
    let mut written = [false; 14];

    // --- narrow filter (used where nothing flat applies) --------------------
    let hev = v128_or(i16x8_gt(absdiff(p1, p0), vh), i16x8_gt(absdiff(q1, q0), vh));
    let lo = i16x8_splat(-128);
    let hi = i16x8_splat(127);
    let clip_diff = |v: v128| i16x8_min(i16x8_max(v, lo), hi);
    let clip_px = |v: v128| i16x8_min(i16x8_max(v, i16x8_splat(0)), i16x8_splat(255));
    let f_in = v128_and(clip_diff(i16x8_sub(p1, q1)), hev);
    let f = clip_diff(i16x8_add(
        i16x8_mul(i16x8_splat(3), i16x8_sub(q0, p0)),
        f_in,
    ));
    let f1 = i16x8_shr(i16x8_min(i16x8_add(f, i16x8_splat(4)), hi), 3);
    let f2 = i16x8_shr(i16x8_min(i16x8_add(f, i16x8_splat(3)), hi), 3);
    let n_p0 = clip_px(i16x8_add(p0, f2));
    let n_q0 = clip_px(i16x8_sub(q0, f1));
    let f3 = i16x8_shr(i16x8_add(f1, i16x8_splat(1)), 1);
    let n_p1 = v128_bitselect(p1, clip_px(i16x8_add(p1, f3)), hev);
    let n_q1 = v128_bitselect(q1, clip_px(i16x8_sub(q1, f3)), hev);

    // --- flatness ------------------------------------------------------------
    let mut flat8in = i16x8_splat(0);
    if wd >= 6 {
        let (p2, q2) = (t.p(2), t.q(2));
        flat8in = v128_and(
            v128_and(le_thresh(p2, p0, vf), le_thresh(p1, p0, vf)),
            v128_and(le_thresh(q1, q0, vf), le_thresh(q2, q0, vf)),
        );
        if wd >= 8 {
            let (p3, q3) = (t.p(3), t.q(3));
            flat8in = v128_and(
                flat8in,
                v128_and(le_thresh(p3, p0, vf), le_thresh(q3, q0, vf)),
            );
        }
    }
    let mut flat8out = i16x8_splat(0);
    if wd >= 16 {
        let (p6, p5, p4, q4, q5, q6) = (t.p(6), t.p(5), t.p(4), t.q(4), t.q(5), t.q(6));
        flat8out = v128_and(
            v128_and(
                v128_and(le_thresh(p6, p0, vf), le_thresh(p5, p0, vf)),
                le_thresh(p4, p0, vf),
            ),
            v128_and(
                v128_and(le_thresh(q4, q0, vf), le_thresh(q5, q0, vf)),
                le_thresh(q6, q0, vf),
            ),
        );
    }

    let m16 = if wd >= 16 {
        v128_and(fm, v128_and(flat8out, flat8in))
    } else {
        i16x8_splat(0)
    };
    let m8 = if wd >= 8 {
        v128_andnot(v128_and(fm, flat8in), m16)
    } else {
        i16x8_splat(0)
    };
    let m6 = if wd == 6 {
        v128_and(fm, flat8in)
    } else {
        i16x8_splat(0)
    };
    let m_narrow = v128_andnot(fm, v128_or(m16, v128_or(m8, m6)));

    // Narrow results first (lowest priority), then overwrite with the wide ones.
    let mut sel = |idx: usize, val: v128, mask: v128, out: &mut [v128; 14]| {
        out[idx] = v128_bitselect(val, out[idx], mask);
        written[idx] = true;
    };
    sel(P - 1, n_p1, m_narrow, &mut out);
    sel(P, n_p0, m_narrow, &mut out);
    sel(Q, n_q0, m_narrow, &mut out);
    sel(Q + 1, n_q1, m_narrow, &mut out);

    if wd == 6 {
        let (p2, q2) = (t.p(2), t.q(2));
        let two = |v: v128| i16x8_shl(v, 1);
        sel(
            P - 1,
            shr_rnd(sum(&[p2, two(p2), two(p1), two(p0), q0]), 4, 3),
            m6,
            &mut out,
        );
        sel(
            P,
            shr_rnd(sum(&[p2, two(p1), two(p0), two(q0), q1]), 4, 3),
            m6,
            &mut out,
        );
        sel(
            Q,
            shr_rnd(sum(&[p1, two(p0), two(q0), two(q1), q2]), 4, 3),
            m6,
            &mut out,
        );
        sel(
            Q + 1,
            shr_rnd(sum(&[p0, two(q0), two(q1), two(q2), q2]), 4, 3),
            m6,
            &mut out,
        );
    }
    if wd >= 8 {
        let (p2, p3, q2, q3) = (t.p(2), t.p(3), t.q(2), t.q(3));
        let two = |v: v128| i16x8_shl(v, 1);
        sel(
            P - 2,
            shr_rnd(sum(&[p3, p3, p3, two(p2), p1, p0, q0]), 4, 3),
            m8,
            &mut out,
        );
        sel(
            P - 1,
            shr_rnd(sum(&[p3, p3, p2, two(p1), p0, q0, q1]), 4, 3),
            m8,
            &mut out,
        );
        sel(
            P,
            shr_rnd(sum(&[p3, p2, p1, two(p0), q0, q1, q2]), 4, 3),
            m8,
            &mut out,
        );
        sel(
            Q,
            shr_rnd(sum(&[p2, p1, p0, two(q0), q1, q2, q3]), 4, 3),
            m8,
            &mut out,
        );
        sel(
            Q + 1,
            shr_rnd(sum(&[p1, p0, q0, two(q1), q2, q3, q3]), 4, 3),
            m8,
            &mut out,
        );
        sel(
            Q + 2,
            shr_rnd(sum(&[p0, q0, q1, two(q2), q3, q3, q3]), 4, 3),
            m8,
            &mut out,
        );
    }
    if wd >= 16 {
        let (p6, p5, p4, p3, p2) = (t.p(6), t.p(5), t.p(4), t.p(3), t.p(2));
        let (q2, q3, q4, q5, q6) = (t.q(2), t.q(3), t.q(4), t.q(5), t.q(6));
        let two = |v: v128| i16x8_shl(v, 1);
        // The 13-tap sums, exactly as the scalar code spells them.
        let r = [
            sum(&[
                p6,
                p6,
                p6,
                p6,
                p6,
                two(p6),
                two(p5),
                two(p4),
                p3,
                p2,
                p1,
                p0,
                q0,
            ]),
            sum(&[
                p6,
                p6,
                p6,
                p6,
                p6,
                two(p5),
                two(p4),
                two(p3),
                p2,
                p1,
                p0,
                q0,
                q1,
            ]),
            sum(&[
                p6,
                p6,
                p6,
                p6,
                p5,
                two(p4),
                two(p3),
                two(p2),
                p1,
                p0,
                q0,
                q1,
                q2,
            ]),
            sum(&[
                p6,
                p6,
                p6,
                p5,
                p4,
                two(p3),
                two(p2),
                two(p1),
                p0,
                q0,
                q1,
                q2,
                q3,
            ]),
            sum(&[
                p6,
                p6,
                p5,
                p4,
                p3,
                two(p2),
                two(p1),
                two(p0),
                q0,
                q1,
                q2,
                q3,
                q4,
            ]),
            sum(&[
                p6,
                p5,
                p4,
                p3,
                p2,
                two(p1),
                two(p0),
                two(q0),
                q1,
                q2,
                q3,
                q4,
                q5,
            ]),
            sum(&[
                p5,
                p4,
                p3,
                p2,
                p1,
                two(p0),
                two(q0),
                two(q1),
                q2,
                q3,
                q4,
                q5,
                q6,
            ]),
            sum(&[
                p4,
                p3,
                p2,
                p1,
                p0,
                two(q0),
                two(q1),
                two(q2),
                q3,
                q4,
                q5,
                q6,
                q6,
            ]),
            sum(&[
                p3,
                p2,
                p1,
                p0,
                q0,
                two(q1),
                two(q2),
                two(q3),
                q4,
                q5,
                q6,
                q6,
                q6,
            ]),
            sum(&[
                p2,
                p1,
                p0,
                q0,
                q1,
                two(q2),
                two(q3),
                two(q4),
                q5,
                q6,
                q6,
                q6,
                q6,
            ]),
            sum(&[
                p1,
                p0,
                q0,
                q1,
                q2,
                two(q3),
                two(q4),
                two(q5),
                q6,
                q6,
                q6,
                q6,
                q6,
            ]),
            sum(&[
                p0,
                q0,
                q1,
                q2,
                q3,
                two(q4),
                two(q5),
                two(q6),
                q6,
                q6,
                q6,
                q6,
                q6,
            ]),
        ];
        for (n, val) in r.into_iter().enumerate() {
            // r[0] is p5 (index P-5) … r[11] is q5 (index Q+5); Q = P+1.
            sel(P - 5 + n, shr_rnd(val, 8, 4), m16, &mut out);
        }
    }

    // --- write back --------------------------------------------------------
    if by_rows {
        for k in 0..14 {
            if !written[k] {
                continue;
            }
            let off = k as isize - Q as isize; // p0 → -1, q0 → 0
            let row = dst + off * strideb;
            let row = &mut *row.slice_mut::<BD>(4);
            let row = row.as_bytes_mut();
            let packed = u8x16_narrow_i16x8(out[k], out[k]);
            // SAFETY: 4 bytes.
            unsafe { v128_store32_lane::<0>(packed, row.as_mut_ptr() as *mut u32) };
        }
    } else {
        let mut packed = [[0u8; 16]; 14];
        for k in 0..14 {
            if written[k] {
                let v = u8x16_narrow_i16x8(out[k], out[k]);
                // SAFETY: 16 bytes.
                unsafe { v128_store(packed[k].as_mut_ptr() as *mut v128, v) };
            }
        }
        for j in 0..4 {
            let row = dst + j as isize * stridea - half;
            let row = &mut *row.slice_mut::<BD>(2 * half);
            let row = row.as_bytes_mut();
            for k in 0..14 {
                if written[k] {
                    // tap k sits at column offset (k - Q); the span starts at -half
                    let col = k + half - Q;
                    row[col] = packed[k][j];
                }
            }
        }
    }
}
