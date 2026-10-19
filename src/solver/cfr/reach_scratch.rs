//! Bounded, reentrant scratch storage for an action's private reach vector.
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};

const MAX_CACHED: usize = 32;
const MAX_COMBOS: usize = 1326;

thread_local! {
    // Only idle buffers live here. Rayon may execute nested work on this
    // thread while a recursive call is waiting; active buffers stay owned
    // by that call, with no RefCell borrow held across recursion.
    static IDLE: RefCell<Vec<Vec<f64>>> = const { RefCell::new(Vec::new()) };
}

pub(super) struct ReachScratch(Vec<f64>);

impl ReachScratch {
    pub(super) fn copy_from(reach: &[f64]) -> Self {
        let mut values = IDLE.with(|idle| idle.borrow_mut().pop().unwrap_or_default());
        values.clear();
        // Avoid geometric growth beyond the cache's capacity bound when
        // successive nodes/solves use different range lengths.
        values.reserve_exact(reach.len());
        values.extend_from_slice(reach);
        Self(values)
    }
}

impl Deref for ReachScratch {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        &self.0
    }
}

impl DerefMut for ReachScratch {
    fn deref_mut(&mut self) -> &mut [f64] {
        &mut self.0
    }
}

impl Drop for ReachScratch {
    fn drop(&mut self) {
        // At most 32 * 1326 * 8 bytes of backing storage per worker, even
        // across independent solves. Oversized core inputs still work but
        // their buffers are freed. TLS teardown must not panic in Drop.
        if self.0.capacity() == 0 || self.0.capacity() > MAX_COMBOS {
            return;
        }
        let _ = IDLE.try_with(|idle| {
            if let Ok(mut idle) = idle.try_borrow_mut() {
                if idle.len() < MAX_CACHED {
                    idle.push(std::mem::take(&mut self.0));
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_replaces_every_value_and_length() {
        IDLE.with(|idle| idle.borrow_mut().clear());
        let mut first = ReachScratch::copy_from(&[7.0; 19]);
        let allocation = first.as_ptr();
        first.fill(f64::NAN);
        drop(first);
        let next = ReachScratch::copy_from(&[0.0, -0.0, 1e-100]);
        assert_eq!(next.as_ptr(), allocation);
        assert_eq!(
            next.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [0.0_f64, -0.0, 1e-100].map(f64::to_bits)
        );
        drop(next);
        let empty = ReachScratch::copy_from(&[]);
        assert!(empty.is_empty());
        drop(empty);
        let larger = ReachScratch::copy_from(&vec![0.25; 1326]);
        assert_eq!(&*larger, &[0.25; 1326]);
    }

    #[test]
    fn idle_storage_is_bounded_and_oversized_inputs_are_not_retained() {
        IDLE.with(|idle| idle.borrow_mut().clear());
        let active: Vec<_> = (0..100)
            .map(|_| ReachScratch::copy_from(&[1.0; 1326]))
            .collect();
        drop(active);
        IDLE.with(|idle| {
            let idle = idle.borrow();
            assert!(idle.len() <= 32);
            assert!(idle.iter().map(|v| v.capacity()).sum::<usize>() <= 32 * 1326);
        });
        let oversized = ReachScratch::copy_from(&[0.5; 1400]);
        assert_eq!(&*oversized, &[0.5; 1400]);
        drop(oversized);
        IDLE.with(|idle| assert!(idle.borrow().iter().all(|v| v.capacity() <= 1326)));
    }

    #[test]
    fn nested_rayon_work_and_unwinding_preserve_active_reaches() {
        fn nested(depth: usize, tag: f64) {
            let expected = vec![tag; depth * 3 + 1];
            let mut active = ReachScratch::copy_from(&expected);
            active[0] *= 0.5;
            if depth > 0 {
                rayon::join(
                    || nested(depth - 1, tag + 1.0),
                    || nested(depth - 1, tag + 100.0),
                );
            }
            assert_eq!(active[0], expected[0] * 0.5);
            assert_eq!(&active[1..], &expected[1..]);
        }
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let survivor = ReachScratch::copy_from(&[0.125; 13]);
                    nested(5, 2.0);
                    let result = std::panic::catch_unwind(|| {
                        let _scratch = ReachScratch::copy_from(&[f64::NAN; 100]);
                        panic!("synthetic recursive failure");
                    });
                    assert!(result.is_err());
                    nested(3, 0.0);
                    assert_eq!(&*survivor, &[0.125; 13]);
                });
        }
    }
}
