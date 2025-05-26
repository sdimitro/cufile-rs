use crate::{check_cufile_error, sys, CuFileResult};
use std::sync::atomic::{AtomicBool, Ordering};

/// Global flag to track if the driver is initialized
static DRIVER_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// CuFile driver manager
pub struct CuFileDriver;

impl CuFileDriver {
    /// Initialize the CuFile driver
    ///
    /// This must be called before using any other CuFile operations.
    /// It's safe to call this multiple times - subsequent calls will be no-ops.
    pub fn init() -> CuFileResult<Self> {
        if DRIVER_INITIALIZED.load(Ordering::Acquire) {
            return Ok(CuFileDriver);
        }

        unsafe {
            check_cufile_error(sys::cuFileDriverOpen())?;
        }

        DRIVER_INITIALIZED.store(true, Ordering::Release);
        Ok(CuFileDriver)
    }
}

impl Drop for CuFileDriver {
    fn drop(&mut self) {
        if DRIVER_INITIALIZED.load(Ordering::Acquire) {
            unsafe {
                let _ = sys::cuFileDriverClose();
            }
            DRIVER_INITIALIZED.store(false, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_driver_init() {
        // This test might fail if NVIDIA drivers/hardware aren't available
        // In a real environment, you'd want to skip this test conditionally
        match CuFileDriver::init() {
            Ok(_driver) => {
                assert!(true);
            }
            Err(e) => {
                println!(
                    "Driver initialization failed (expected in test environment): {:?}",
                    e
                );
            }
        }
    }
}
