//! Scalar C type aliases and `errno` values used across the crate.
//!
//! On every target with a libc these come straight from the [`libc`] crate.
//! `wasm32-unknown-unknown` has no libc, so the [`libc`] crate does not define
//! them there; the aliases are nonetheless the same on every platform
//! (`ptrdiff_t`/`intptr_t` are `isize`, `uintptr_t` is `usize`), and the
//! `errno` values follow Linux, which is what [`Dav1dResult`] callers expect.
//!
//! [`Dav1dResult`]: crate::error::Dav1dResult

#![allow(non_camel_case_types)]

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use libc::{
    intptr_t, off_t, ptrdiff_t, uintptr_t, EAGAIN, EINVAL, ENOENT, ENOMEM, ENOPROTOOPT, ERANGE,
};

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod wasm32_unknown {
    use std::ffi::c_int;

    pub type ptrdiff_t = isize;
    pub type intptr_t = isize;
    pub type uintptr_t = usize;
    pub type off_t = i64;

    pub const ENOENT: c_int = 2;
    pub const EAGAIN: c_int = 11;
    pub const ENOMEM: c_int = 12;
    pub const EINVAL: c_int = 22;
    pub const ERANGE: c_int = 34;
    pub const ENOPROTOOPT: c_int = 92;
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub use wasm32_unknown::*;
