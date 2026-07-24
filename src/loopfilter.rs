#![deny(unsafe_op_in_unsafe_fn)]

use std::cmp;
use std::ffi::c_int;

use libc::ptrdiff_t;
use strum::FromRepr;

use crate::align::{Align16, AlignedVec2};
use crate::cpu::CpuFlags;
use crate::disjoint_mut::DisjointMut;
use crate::ffi_safe::FFISafe;
#[cfg(all(
    feature = "asm",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64"
    )
))]
use crate::include::common::bitdepth::BPC;
use crate::include::common::bitdepth::{AsPrimitive, BitDepth, DynPixel};
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::{
    FFISafeRav1dPictureDataComponentOffset, Rav1dPictureDataComponentOffset,
};
use crate::internal::Rav1dFrameData;
use crate::lf_mask::Av1FilterLUT;
use crate::strided::Strided as _;
use crate::with_offset::WithOffset;
use crate::wrap_fn_ptr::wrap_fn_ptr;

wrap_fn_ptr!(unsafe extern "C" fn loopfilter_sb(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: FFISafeRav1dPictureDataComponentOffset,
    _lvl: WithOffset<*const FFISafe<DisjointMut<AlignedVec2<u8>>>>,
) -> ());

// The exact 8-bit assembly ABI: `decl_loopfilter_sb_fn` (`src/loopfilter.h`) with an empty
// `HIGHBD_DECL_SUFFIX`. Only `LoopFilterSbFn` may name it.
#[cfg(all(
    feature = "asm",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64"
    )
))]
wrap_fn_ptr!(unsafe extern "C" fn loopfilter_sb_asm8(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
) -> ());

// The same with `HIGHBD_DECL_SUFFIX` expanded to `, const int bitdepth_max`.
#[cfg(all(
    feature = "asm",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64"
    )
))]
wrap_fn_ptr!(unsafe extern "C" fn loopfilter_sb_asm16(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
) -> ());

/// One loop-filter function slot.
///
/// The live arm is not encoded per slot; it is given by the `LoopFilterKind` that
/// [`Rav1dLoopFilterDSPContext::new`] returned together with the slots.
/// Every arm must stay pointer-sized so that a slot is exactly 8 bytes:
/// a fully tagged 16-byte slot measured +0.095% cycles and was rejected.
#[derive(Clone, Copy)]
pub(crate) union LoopFilterSbFn {
    rust: loopfilter_sb::Fn,
    #[cfg(all(
        feature = "asm",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "arm",
            target_arch = "aarch64"
        )
    ))]
    asm8: loopfilter_sb_asm8::Fn,
    #[cfg(all(
        feature = "asm",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "arm",
            target_arch = "aarch64"
        )
    ))]
    asm16: loopfilter_sb_asm16::Fn,
}

/// Which [`LoopFilterSbFn`] arm every slot of a [`Rav1dLoopFilterDSPContext`] holds.
///
/// This is the tag of an untagged union that is read on the dispatch path, so it is a
/// soundness invariant that it describes the slots it was returned with:
/// * `Kind::Rust`: all four slots hold their `rust` arm.
/// * `Kind::Asm`: all four slots hold the `asm8` or `asm16` arm matching the bit depth
///   the slots were constructed for.
///
/// The invariant is structural, not conventional: [`Rav1dLoopFilterDSPContext::new`] is
/// the only way to obtain either half, and it derives both from the same branch.
/// `Kind` and the `RUST`/`ASM` constants are private to this module, so no other module
/// can construct a tag, and the slots are only reachable through `&`-accessors, so no
/// other module can alter them.
#[derive(Clone, Copy)]
pub(crate) struct LoopFilterKind(Kind);

#[derive(Clone, Copy)]
enum Kind {
    Rust,
    #[cfg(all(
        feature = "asm",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "arm",
            target_arch = "aarch64"
        )
    ))]
    Asm,
}

impl LoopFilterKind {
    const RUST: Self = Self(Kind::Rust);

    #[cfg(all(
        feature = "asm",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "arm",
            target_arch = "aarch64"
        )
    ))]
    const ASM: Self = Self(Kind::Asm);
}

#[cfg(all(
    feature = "asm",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64"
    )
))]
macro_rules! loopfilter_asm_fn {
    ($BD:ty, $name:ident, $asm:ident) => {{
        use paste::paste;

        paste! {
            match $BD::BPC {
                BPC::BPC8 => LoopFilterSbFn {
                    asm8: loopfilter_sb_asm8::decl_fn!(
                        fn [<dav1d_ $name _8bpc_ $asm>]
                    ),
                },
                BPC::BPC16 => LoopFilterSbFn {
                    asm16: loopfilter_sb_asm16::decl_fn!(
                        fn [<dav1d_ $name _16bpc_ $asm>]
                    ),
                },
            }
        }
    }};
}

impl LoopFilterSbFn {
    /// The union arm read here is selected by `f.dsp`'s [`LoopFilterKind`] and, for the
    /// assembly arms, by the call-site `BD::BPC`, while the slots were written at
    /// construction time. Two invariants make that sound, and both hold for every caller:
    ///
    /// * `self` is one of `f.dsp`'s own four slots, so the tag describes it
    ///   (`debug_assert`ed below).
    /// * `BD` agrees with the bit depth `f.dsp` was constructed for. The context is
    ///   selected from `seq_hdr.hbd` (`src/decode.rs`, via
    ///   `Rav1dBitDepthDSPContext::get`), while `BD` is derived from `f.cur.p.bpc`
    ///   through `f.bitdepth_max` (`src/decode.rs`, via `Rav1dFrameData::bd_fn`);
    ///   both reduce to the same 8-bit vs. high-bitdepth split.
    pub(crate) fn call<BD: BitDepth>(
        &self,
        f: &Rav1dFrameData,
        dst: Rav1dPictureDataComponentOffset,
        mask: &[u32; 3],
        lvl: WithOffset<&DisjointMut<AlignedVec2<u8>>>,
        w: usize,
    ) {
        // Debug builds only; the release vector suites do not execute this.
        #[cfg(debug_assertions)]
        assert!(f.dsp.lf().contains(self));

        let dst_ptr = dst.as_mut_ptr::<BD>().cast();
        let stride = dst.stride();
        assert!(lvl.offset <= lvl.data.len());
        // SAFETY: `lvl.offset` is in bounds, just checked above.
        let lvl_ptr = unsafe { lvl.data.as_mut_ptr().add(lvl.offset) };
        let lvl_ptr = lvl_ptr.cast::<[u8; 4]>();
        let b4_stride = f.b4_stride;
        let lut = &f.lf.lim_lut;
        let w = w as c_int;
        let bd = f.bitdepth_max;
        match f.dsp.lf_kind().0 {
            Kind::Rust => {
                // SAFETY: `LoopFilterKind`'s invariant makes `rust` the active arm.
                let fallback = unsafe { self.rust };
                let dst = dst.into_ffi_safe();
                let lvl = lvl.into_ffi_safe();
                // SAFETY: The fallback reconstructs the checked Rust values passed here.
                unsafe {
                    fallback.get()(
                        dst_ptr, stride, mask, lvl_ptr, b4_stride, lut, w, bd, dst, lvl,
                    )
                }
            }
            #[cfg(all(
                feature = "asm",
                any(
                    target_arch = "x86",
                    target_arch = "x86_64",
                    target_arch = "arm",
                    target_arch = "aarch64"
                )
            ))]
            Kind::Asm => match BD::BPC {
                BPC::BPC8 => {
                    // SAFETY: `LoopFilterKind`'s invariant makes `asm8` the active arm.
                    let asm = unsafe { self.asm8 };
                    // SAFETY: This is an 8-bit assembly implementation.
                    unsafe { asm.get()(dst_ptr, stride, mask, lvl_ptr, b4_stride, lut, w) }
                }
                BPC::BPC16 => {
                    // SAFETY: `LoopFilterKind`'s invariant makes `asm16` the active arm.
                    let asm = unsafe { self.asm16 };
                    // SAFETY: This is a high-bitdepth assembly implementation.
                    unsafe { asm.get()(dst_ptr, stride, mask, lvl_ptr, b4_stride, lut, w, bd) }
                }
            },
        }
    }

    const fn default<BD: BitDepth, const HV: usize, const YUV: usize>() -> Self {
        Self {
            rust: loopfilter_sb::Fn::new(loop_filter_sb128_c_erased::<BD, { HV }, { YUV }>),
        }
    }

    /// The stored pointer as a bare address, for debug checks only.
    #[cfg(debug_assertions)]
    fn addr(&self) -> usize {
        // SAFETY: Every arm is a non-null `extern "C"` `fn` ptr of the same size,
        // and the result is only compared as an address, never called through.
        let fn_ptr = unsafe { self.rust };
        *fn_ptr.get() as usize
    }
}

pub(crate) struct LoopFilterHVDSPContext {
    pub h: LoopFilterSbFn,
    pub v: LoopFilterSbFn,
}

pub(crate) struct LoopFilterYUVDSPContext {
    pub y: LoopFilterHVDSPContext,
    pub uv: LoopFilterHVDSPContext,
}

pub(crate) struct Rav1dLoopFilterDSPContext {
    /// Private: the slots are only handed out behind a `&`, so that no other module can
    /// alter them out from under the [`LoopFilterKind`] tag they were constructed with.
    loop_filter_sb: LoopFilterYUVDSPContext,
}

#[inline(never)]
fn loop_filter<BD: BitDepth>(
    dst: Rav1dPictureDataComponentOffset,
    e: u8,
    i: u8,
    h: u8,
    stridea: ptrdiff_t,
    strideb: ptrdiff_t,
    wd: c_int,
    bd: BD,
) {
    let bitdepth_min_8 = bd.bitdepth() - 8;
    let [f, e, i, h] = [1, e, i, h].map(|n| (n as i32) << bitdepth_min_8);

    for idx in 0..4 {
        let dst = dst + (idx * stridea);
        let dst = |stride_index: isize| (dst + (strideb * stride_index)).index_mut::<BD>();

        let get_dst = |stride_index| (*dst(stride_index)).as_::<i32>();
        let set_dst = |stride_index, pixel: i32| {
            *dst(stride_index) = pixel.as_::<BD::Pixel>();
        };
        let set_dst_clipped = |stride_index, pixel: i32| {
            *dst(stride_index) = bd.iclip_pixel(pixel);
        };

        let mut p6 = 0;
        let mut p5 = 0;
        let mut p4 = 0;
        let mut p3 = 0;
        let mut p2 = 0;
        let p1 = get_dst(-2);
        let p0 = get_dst(-1);
        let q0 = get_dst(0);
        let q1 = get_dst(1);
        let mut q2 = 0;
        let mut q3 = 0;
        let mut q4 = 0;
        let mut q5 = 0;
        let mut q6 = 0;
        let mut flat8out = false;
        let mut flat8in = false;

        let mut fm = (p1 - p0).abs() <= i
            && (q1 - q0).abs() <= i
            && (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= e;

        if wd > 4 {
            p2 = get_dst(-3);
            q2 = get_dst(2);

            fm &= (p2 - p1).abs() <= i && (q2 - q1).abs() <= i;

            if wd > 6 {
                p3 = get_dst(-4);
                q3 = get_dst(3);

                fm &= (p3 - p2).abs() <= i && (q3 - q2).abs() <= i;
            }
        }
        if !fm {
            continue;
        }

        if wd >= 16 {
            p6 = get_dst(-7);
            p5 = get_dst(-6);
            p4 = get_dst(-5);
            q4 = get_dst(4);
            q5 = get_dst(5);
            q6 = get_dst(6);

            flat8out = (p6 - p0).abs() <= f
                && (p5 - p0).abs() <= f
                && (p4 - p0).abs() <= f
                && (q4 - q0).abs() <= f
                && (q5 - q0).abs() <= f
                && (q6 - q0).abs() <= f;
        }

        if wd >= 6 {
            flat8in = (p2 - p0).abs() <= f
                && (p1 - p0).abs() <= f
                && (q1 - q0).abs() <= f
                && (q2 - q0).abs() <= f;
        }

        if wd >= 8 {
            flat8in &= (p3 - p0).abs() <= f && (q3 - q0).abs() <= f;
        }

        if wd >= 16 && flat8out && flat8in {
            set_dst(
                -6,
                p6 + p6 + p6 + p6 + p6 + p6 * 2 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + 8 >> 4,
            );
            set_dst(
                -5,
                p6 + p6 + p6 + p6 + p6 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + 8 >> 4,
            );
            set_dst(
                -4,
                p6 + p6 + p6 + p6 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + 8 >> 4,
            );
            set_dst(
                -3,
                p6 + p6 + p6 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + 8 >> 4,
            );
            set_dst(
                -2,
                p6 + p6 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + 8 >> 4,
            );
            set_dst(
                -1,
                p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + 8 >> 4,
            );
            set_dst(
                0,
                p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + 8 >> 4,
            );
            set_dst(
                1,
                p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 + q6 + 8 >> 4,
            );
            set_dst(
                2,
                p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 + q6 + q6 + 8 >> 4,
            );
            set_dst(
                3,
                p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 + q6 + q6 + q6 + 8 >> 4,
            );
            set_dst(
                4,
                p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 + q6 + q6 + q6 + q6 + 8 >> 4,
            );
            set_dst(
                5,
                p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 2 + q6 + q6 + q6 + q6 + q6 + 8 >> 4,
            );
        } else if wd >= 8 && flat8in {
            set_dst(-3, p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4 >> 3);
            set_dst(-2, p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4 >> 3);
            set_dst(-1, p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4 >> 3);
            set_dst(0, p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4 >> 3);
            set_dst(1, p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4 >> 3);
            set_dst(2, p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4 >> 3);
        } else if wd == 6 && flat8in {
            set_dst(-2, p2 + 2 * p2 + 2 * p1 + 2 * p0 + q0 + 4 >> 3);
            set_dst(-1, p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4 >> 3);
            set_dst(0, p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4 >> 3);
            set_dst(1, p0 + 2 * q0 + 2 * q1 + 2 * q2 + q2 + 4 >> 3);
        } else {
            let hev = (p1 - p0).abs() > h || (q1 - q0).abs() > h;

            fn iclip_diff(v: c_int, bitdepth_min_8: u8) -> i32 {
                iclip(
                    v,
                    -128 * (1 << bitdepth_min_8),
                    128 * (1 << bitdepth_min_8) - 1,
                )
            }

            if hev {
                let f = iclip_diff(p1 - q1, bitdepth_min_8);
                let f = iclip_diff(3 * (q0 - p0) + f, bitdepth_min_8);

                let f1 = cmp::min(f + 4, (128 << bitdepth_min_8) - 1) >> 3;
                let f2 = cmp::min(f + 3, (128 << bitdepth_min_8) - 1) >> 3;

                set_dst_clipped(-1, p0 + f2);
                set_dst_clipped(0, q0 - f1);
            } else {
                let f = iclip_diff(3 * (q0 - p0), bitdepth_min_8);

                let f1 = cmp::min(f + 4, (128 << bitdepth_min_8) - 1) >> 3;
                let f2 = cmp::min(f + 3, (128 << bitdepth_min_8) - 1) >> 3;

                set_dst_clipped(-1, p0 + f2);
                set_dst_clipped(0, q0 - f1);

                let f = (f1 + 1) >> 1;
                set_dst_clipped(-2, p1 + f);
                set_dst_clipped(1, q1 - f);
            }
        }
    }
}

#[derive(FromRepr)]
enum HV {
    H,
    V,
}

#[derive(FromRepr)]
enum YUV {
    Y,
    UV,
}

fn loop_filter_sb128_rust<BD: BitDepth, const HV: usize, const YUV: usize>(
    mut dst: Rav1dPictureDataComponentOffset,
    vmask: &[u32; 3],
    mut lvl: WithOffset<&DisjointMut<AlignedVec2<u8>>>,
    b4_stride: usize,
    lut: &Align16<Av1FilterLUT>,
    _wh: c_int,
    bd: BD,
) {
    let hv = HV::from_repr(HV).unwrap();
    let yuv = YUV::from_repr(YUV).unwrap();

    let stride = dst.pixel_stride::<BD>();
    let (stridea, strideb) = match hv {
        HV::H => (stride, 1),
        HV::V => (1, stride),
    };
    let (b4_stridea, b4_strideb) = match hv {
        HV::H => (b4_stride, 1),
        HV::V => (1, b4_stride),
    };

    let vm = match yuv {
        YUV::Y => vmask[0] | vmask[1] | vmask[2],
        YUV::UV => vmask[0] | vmask[1],
    };
    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        'block: {
            if vm & xy == 0 {
                break 'block;
            }
            let l = *lvl.data.index(lvl.offset);
            let l = if l != 0 {
                l
            } else {
                let lvl = lvl - 4 * b4_strideb;
                *lvl.data.index(lvl.offset)
            };
            if l == 0 {
                break 'block;
            }
            let h = l >> 4;
            let e = lut.0.e[l as usize];
            let i = lut.0.i[l as usize];
            let idx = match yuv {
                YUV::Y => {
                    let idx = if vmask[2] & xy != 0 {
                        2
                    } else {
                        (vmask[1] & xy != 0) as c_int
                    };
                    4 << idx
                }
                YUV::UV => {
                    let idx = (vmask[1] & xy != 0) as c_int;
                    4 + 2 * idx
                }
            };
            loop_filter(dst, e, i, h, stridea, strideb, idx, bd);
        }
        xy <<= 1;
        dst += 4 * stridea;
        lvl += 4 * b4_stridea;
    }
}

/// # Safety
///
/// Must be called by [`LoopFilterSbFn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn loop_filter_sb128_c_erased<BD: BitDepth, const HV: usize, const YUV: usize>(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    vmask: &[u32; 3],
    _lvl_ptr: *const [u8; 4],
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    wh: c_int,
    bitdepth_max: c_int,
    dst: FFISafeRav1dPictureDataComponentOffset,
    lvl: WithOffset<*const FFISafe<DisjointMut<AlignedVec2<u8>>>>,
) {
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `LoopFilterSbFn::call`.
    let dst = unsafe { FFISafe::from_with_offset(dst) };
    // SAFETY: Was passed as `WithOffset::into_ffi_safe(_)` in `LoopFilterSbFn::call`.
    let lvl = unsafe { FFISafe::from_with_offset(lvl) };
    let b4_stride = b4_stride as usize;
    let bd = BD::from_c(bitdepth_max);
    loop_filter_sb128_rust::<BD, { HV }, { YUV }>(dst, vmask, lvl, b4_stride, lut, wh, bd)
}

impl Rav1dLoopFilterDSPContext {
    #[inline(always)]
    pub(crate) fn loop_filter_sb(&self) -> &LoopFilterYUVDSPContext {
        &self.loop_filter_sb
    }

    /// Whether `slot` is one of this context's four slots, i.e. whether this context's
    /// [`LoopFilterKind`] describes it. Debug checks only.
    #[cfg(debug_assertions)]
    fn contains(&self, slot: &LoopFilterSbFn) -> bool {
        let sb = &self.loop_filter_sb;
        [&sb.y.h, &sb.y.v, &sb.uv.h, &sb.uv.v]
            .into_iter()
            .any(|s| std::ptr::eq(s, slot))
    }

    /// Verifies that `kind` describes what [`Self::init`] actually wrote: an assembly tag
    /// requires that all four fallback slots were overwritten, and a Rust tag requires
    /// that none of them were. This is the failure mode a divergence between the tag and
    /// the assembly thresholds would produce, and it would be UB on the dispatch path.
    ///
    /// This only runs under `debug_assertions`; release builds, including the release
    /// vector suites, do not execute it.
    #[cfg(debug_assertions)]
    pub(crate) fn debug_assert_kind<BD: BitDepth>(&self, kind: LoopFilterKind) {
        let default = Self::default::<BD>();
        let sb = &self.loop_filter_sb;
        let fb = &default.loop_filter_sb;
        for (slot, fallback) in [
            (&sb.y.h, &fb.y.h),
            (&sb.y.v, &fb.y.v),
            (&sb.uv.h, &fb.uv.h),
            (&sb.uv.v, &fb.uv.v),
        ] {
            match kind.0 {
                Kind::Rust => assert_eq!(slot.addr(), fallback.addr()),
                #[cfg(all(
                    feature = "asm",
                    any(
                        target_arch = "x86",
                        target_arch = "x86_64",
                        target_arch = "arm",
                        target_arch = "aarch64"
                    )
                ))]
                Kind::Asm => assert_ne!(slot.addr(), fallback.addr()),
            }
        }
    }

    const fn default<BD: BitDepth>() -> Self {
        use HV::*;
        use YUV::*;
        Self {
            loop_filter_sb: LoopFilterYUVDSPContext {
                y: LoopFilterHVDSPContext {
                    h: LoopFilterSbFn::default::<BD, { H as _ }, { Y as _ }>(),
                    v: LoopFilterSbFn::default::<BD, { V as _ }, { Y as _ }>(),
                },
                uv: LoopFilterHVDSPContext {
                    h: LoopFilterSbFn::default::<BD, { H as _ }, { UV as _ }>(),
                    v: LoopFilterSbFn::default::<BD, { V as _ }, { UV as _ }>(),
                },
            },
        }
    }

    /// The `SSSE3` threshold is tested here only: the tag returned alongside the slots
    /// comes from the same branch that assigns them, so the two cannot diverge.
    #[cfg(all(feature = "asm", any(target_arch = "x86", target_arch = "x86_64")))]
    #[inline(always)]
    const fn init_x86<BD: BitDepth>(mut self, flags: CpuFlags) -> (Self, LoopFilterKind) {
        if !flags.contains(CpuFlags::SSSE3) {
            return (self, LoopFilterKind::RUST);
        }

        self.loop_filter_sb.y.h = loopfilter_asm_fn!(BD, lpf_h_sb_y, ssse3);
        self.loop_filter_sb.y.v = loopfilter_asm_fn!(BD, lpf_v_sb_y, ssse3);
        self.loop_filter_sb.uv.h = loopfilter_asm_fn!(BD, lpf_h_sb_uv, ssse3);
        self.loop_filter_sb.uv.v = loopfilter_asm_fn!(BD, lpf_v_sb_uv, ssse3);

        #[cfg(target_arch = "x86_64")]
        {
            if !flags.contains(CpuFlags::AVX2) {
                return (self, LoopFilterKind::ASM);
            }

            self.loop_filter_sb.y.h = loopfilter_asm_fn!(BD, lpf_h_sb_y, avx2);
            self.loop_filter_sb.y.v = loopfilter_asm_fn!(BD, lpf_v_sb_y, avx2);
            self.loop_filter_sb.uv.h = loopfilter_asm_fn!(BD, lpf_h_sb_uv, avx2);
            self.loop_filter_sb.uv.v = loopfilter_asm_fn!(BD, lpf_v_sb_uv, avx2);

            if !flags.contains(CpuFlags::AVX512ICL) {
                return (self, LoopFilterKind::ASM);
            }

            self.loop_filter_sb.y.v = loopfilter_asm_fn!(BD, lpf_v_sb_y, avx512icl);
            self.loop_filter_sb.uv.v = loopfilter_asm_fn!(BD, lpf_v_sb_uv, avx512icl);

            if !flags.contains(CpuFlags::SLOW_GATHER) {
                self.loop_filter_sb.y.h = loopfilter_asm_fn!(BD, lpf_h_sb_y, avx512icl);
                self.loop_filter_sb.uv.h = loopfilter_asm_fn!(BD, lpf_h_sb_uv, avx512icl);
            }
        }

        (self, LoopFilterKind::ASM)
    }

    /// The `NEON` threshold is tested here only; see [`Self::init_x86`].
    #[cfg(all(feature = "asm", any(target_arch = "arm", target_arch = "aarch64")))]
    #[inline(always)]
    const fn init_arm<BD: BitDepth>(mut self, flags: CpuFlags) -> (Self, LoopFilterKind) {
        if !flags.contains(CpuFlags::NEON) {
            return (self, LoopFilterKind::RUST);
        }

        self.loop_filter_sb.y.h = loopfilter_asm_fn!(BD, lpf_h_sb_y, neon);
        self.loop_filter_sb.y.v = loopfilter_asm_fn!(BD, lpf_v_sb_y, neon);
        self.loop_filter_sb.uv.h = loopfilter_asm_fn!(BD, lpf_h_sb_uv, neon);
        self.loop_filter_sb.uv.v = loopfilter_asm_fn!(BD, lpf_v_sb_uv, neon);

        (self, LoopFilterKind::ASM)
    }

    #[inline(always)]
    const fn init<BD: BitDepth>(self, flags: CpuFlags) -> (Self, LoopFilterKind) {
        #[cfg(feature = "asm")]
        {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                return self.init_x86::<BD>(flags);
            }
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            {
                return self.init_arm::<BD>(flags);
            }
        }

        #[allow(unreachable_code)] // Reachable on some #[cfg]s.
        {
            let _ = flags;
            (self, LoopFilterKind::RUST)
        }
    }

    /// Constructs the function slots together with the [`LoopFilterKind`] that describes
    /// them, both from the same branch of [`Self::init`].
    ///
    /// This is the only way to obtain either half: [`Self::default`] is private and
    /// `LoopFilterKind` is not constructible outside this module, so a tag cannot be
    /// paired with slots it does not describe without editing this file.
    pub(crate) const fn new<BD: BitDepth>(flags: CpuFlags) -> (Self, LoopFilterKind) {
        Self::default::<BD>().init::<BD>(flags)
    }
}
