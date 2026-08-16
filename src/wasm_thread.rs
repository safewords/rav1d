//! Thread spawning on `wasm32` with the `atomics` target feature.
//!
//! `std::thread::spawn` is unsupported on `wasm32-unknown-unknown`: a wasm
//! thread is a Web Worker running another instance of the same module on the
//! same shared memory, and only the embedder can create one. So the embedder
//! registers a spawner once (before [`rav1d_open`](crate::rav1d_open) with
//! `n_threads > 1`), and `rav1d_open` hands each worker thread's body to it.
//! The spawner must run the body on a new thread of the same instance
//! (module + memory) — wasm-av1 has one that does it with wasm-bindgen and a
//! `Worker`. Everything else the worker threads use (`parking_lot` mutexes
//! and condvars) sits on `memory.atomic.wait`/`notify` and works as-is.
//!
//! The spawner returns as soon as the thread is *requested*: a Worker starts
//! asynchronously and must not be waited for on the spawning thread. The
//! worker threads therefore wait for the context on a condvar rather than
//! being unparked by handle, which also lets `rav1d_open` finish before any
//! of them has started.

use std::sync::OnceLock;

/// The body of one worker thread.
pub type ThreadBody = Box<dyn FnOnce() + Send + 'static>;

/// Starts a thread running `body`, or fails without having started it.
pub type Spawner = fn(ThreadBody) -> Result<(), ()>;

static SPAWNER: OnceLock<Spawner> = OnceLock::new();

/// Register the function that starts worker threads. Only the first call
/// takes effect; returns whether this one did.
pub fn set_thread_spawner(spawner: Spawner) -> bool {
    SPAWNER.set(spawner).is_ok()
}

/// Whether a spawner is registered, i.e. whether `n_threads > 1` can work.
pub fn is_thread_spawner_set() -> bool {
    SPAWNER.get().is_some()
}

pub(crate) fn spawn(body: impl FnOnce() + Send + 'static) -> Result<(), ()> {
    let spawner = SPAWNER.get().ok_or(())?;
    spawner(Box::new(body))
}
