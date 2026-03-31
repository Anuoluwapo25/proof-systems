//! A `no_std`-compatible replacement for [`std::sync::LazyLock`].
//!
//! When the `std` feature is enabled, this delegates directly to
//! [`std::sync::LazyLock`]. In `no_std` mode, initialization is coordinated
//! with two [`AtomicBool`](core::sync::atomic::AtomicBool)s:
//!
//! - `initializing` — claimed via `compare_exchange` by the first thread to
//!   arrive. The winner runs the init function and writes the value.
//! - `initialized` — set with `Release` ordering after the value is written.
//!   All readers `Acquire` this flag before accessing the value.
//!
//! Concurrent callers that lose the race spin on `initialized` via
//! [`core::hint::spin_loop`].

use core::ops::Deref;

/// A thread-safe, lazily-initialized value.
///
/// Constructed with a function (typically `fn() -> T`) so that it can be used
/// in `static` items via [`LazyLock::new`], which is `const`.
///
/// # Examples
///
/// ```
/// use o1_utils::lazy_lock::LazyLock;
///
/// static VALUE: LazyLock<u64> = LazyLock::new(|| 1 + 2);
///
/// assert_eq!(*VALUE, 3);
/// ```
#[cfg(feature = "std")]
pub struct LazyLock<T, F = fn() -> T> {
    inner: std::sync::LazyLock<T, F>,
}

#[cfg(feature = "std")]
impl<T, F: FnOnce() -> T> LazyLock<T, F> {
    pub const fn new(init: F) -> Self {
        Self {
            inner: std::sync::LazyLock::new(init),
        }
    }
}

#[cfg(feature = "std")]
impl<T, F: FnOnce() -> T> Deref for LazyLock<T, F> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.inner
    }
}

/// A thread-safe, lazily-initialized value (no-std fallback).
///
/// See the [module-level documentation](self) for the synchronization protocol.
#[cfg(not(feature = "std"))]
pub struct LazyLock<T, F = fn() -> T> {
    state: core::sync::atomic::AtomicU8,
    data: core::cell::UnsafeCell<core::mem::MaybeUninit<T>>,
    init: core::cell::UnsafeCell<Option<F>>,
}

#[cfg(not(feature = "std"))]
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Uninitialized = 0,
    Initializing = 1,
    Initialized = 2,
    Poisoned = 3,
}

#[cfg(not(feature = "std"))]
impl State {
    #[inline]
    fn from_u8(value: u8) -> Self {
        match value {
            0 => State::Uninitialized,
            1 => State::Initializing,
            2 => State::Initialized,
            3 => State::Poisoned,
            _ => unreachable!("Invalid LazyLock state"),
        }
    }
}

// SAFETY: Access to `data` and `init` is synchronized through the `state` atomic.
// Only the thread that transitions the state from `Uninitialized` to `Initializing`
// writes to `data` and `init`. All other threads spin until the state becomes
// `Initialized`, after which `data` is read-only.
#[cfg(not(feature = "std"))]
#[allow(unsafe_code)]
unsafe impl<T: Send + Sync, F: Send> Sync for LazyLock<T, F> {}
#[cfg(not(feature = "std"))]
#[allow(unsafe_code)]
unsafe impl<T: Send, F: Send> Send for LazyLock<T, F> {}

#[cfg(not(feature = "std"))]
impl<T, F: FnOnce() -> T> LazyLock<T, F> {
    pub const fn new(init: F) -> Self {
        Self {
            state: core::sync::atomic::AtomicU8::new(State::Uninitialized as u8),
            data: core::cell::UnsafeCell::new(core::mem::MaybeUninit::uninit()),
            init: core::cell::UnsafeCell::new(Some(init)),
        }
    }

    #[allow(unsafe_code)]
    fn force(&self) -> &T {
        use core::sync::atomic::Ordering;

        // Fast path: Check if already initialized.
        if self.state.load(Ordering::Acquire) == State::Initialized as u8 {
            // SAFETY: `state` is Initialized, meaning `data` has been fully
            // written, and `Acquire` ordering synchronizes with the `Release`
            // store in the initializing thread.
            return unsafe { (*self.data.get()).assume_init_ref() };
        }

        // Slow path: Spin-lock and initialization
        loop {
            match self.state.compare_exchange_weak(
                State::Uninitialized as u8,
                State::Initializing as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // We won the race — initialize the value.
                    // Guard to catch panics in the init closure and poison the lock.
                    struct PanicGuard<'a>(&'a core::sync::atomic::AtomicU8);
                    impl Drop for PanicGuard<'_> {
                        fn drop(&mut self) {
                            self.0.store(State::Poisoned as u8, Ordering::Release);
                        }
                    }
                    let guard = PanicGuard(&self.state);

                    // SAFETY: We are the only thread past the compare-exchange into Initializing,
                    // so we have exclusive access to `data` and `init`.
                    unsafe {
                        let init_fn = (*self.init.get())
                            .take()
                            .expect("LazyLock init function missing");
                        (*self.data.get()).write(init_fn());
                    }

                    // Success: Update state and prevent the guard from poisoning the lock.
                    self.state
                        .store(State::Initialized as u8, Ordering::Release);
                    core::mem::forget(guard); // Disarm the panic guard

                    // SAFETY: We just fully initialized the data.
                    return unsafe { (*self.data.get()).assume_init_ref() };
                }
                Err(current_state_u8) => match State::from_u8(current_state_u8) {
                    State::Initialized => {
                        // Another thread finished initializing while we were spinning.
                        // SAFETY: `state` is Initialized, `Acquire` ordering guarantees visibility.
                        return unsafe { (*self.data.get()).assume_init_ref() };
                    }
                    State::Initializing => {
                        // Another thread is actively initializing — spin until done.
                        core::hint::spin_loop();
                    }
                    State::Poisoned => {
                        // The initializing thread panicked.
                        core::panic!("LazyLock instance has previously been poisoned");
                    }
                    State::Uninitialized => {
                        // Spurious failure from `compare_exchange_weak`, loop immediately.
                        continue;
                    }
                },
            }
        }
    }
}

#[cfg(not(feature = "std"))]
impl<T, F> core::ops::Drop for LazyLock<T, F> {
    fn drop(&mut self) {
        // We have an exclusive reference (`&mut self`), so no other thread can be
        // accessing this lock. We can safely bypass atomic loads and use `get_mut()`.
        if *self.state.get_mut() == State::Initialized as u8 {
            // SAFETY: If the state is Initialized, `data` contains a valid, fully
            // initialized `T`. We must drop it to prevent memory leaks.
            #[allow(unsafe_code)]
            unsafe {
                (*self.data.get()).assume_init_drop();
            }
        }
    }
}

#[cfg(not(feature = "std"))]
impl<T, F: FnOnce() -> T> Deref for LazyLock<T, F> {
    type Target = T;

    fn deref(&self) -> &T {
        self.force()
    }
}

#[cfg(test)]
mod tests {
    use super::LazyLock;
    extern crate std;

    #[test]
    fn lazy_lock_panic() {
        static VALUE: LazyLock<u64> = LazyLock::new(|| {
            panic!("test_lazy_lock_panic");
        });

        std::thread::scope(|s| {
            let error_counts = (0..4)
                .map(|_| {
                    s.spawn(|| {
                        assert_eq!(*VALUE, 3);
                    })
                })
                .map(|thread| thread.join().unwrap_err())
                .map(|err| *err.downcast_ref::<&'static str>().unwrap())
                .fold(std::collections::HashMap::new(), |mut acc, err| {
                    *acc.entry(err).or_insert(0) += 1;
                    acc
                });

            assert_eq!(error_counts.get("test_lazy_lock_panic").copied(), Some(1));
            assert_eq!(
                error_counts
                    .get("LazyLock instance has previously been poisoned")
                    .copied(),
                Some(3)
            );
        });
    }

    #[test]
    fn lazy_lock_success() {
        static VALUE: LazyLock<u64> = LazyLock::new(|| 3);

        std::thread::scope(|s| {
            let threads = (0..4)
                .map(|_| {
                    s.spawn(|| {
                        assert_eq!(*VALUE, 3);
                    })
                })
                .collect::<alloc::vec::Vec<_>>();

            for thread in threads {
                thread.join().unwrap();
            }

            assert_eq!(*VALUE, 3);
        });
    }
}
