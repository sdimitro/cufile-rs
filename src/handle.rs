use crate::{check_cufile_error, sys, CuFileResult};
use std::fs::File;
use std::os::raw::{c_int, c_void};
use std::os::unix::io::AsRawFd;
use std::ptr;

/// A registered CuFile handle for performing GPU-direct storage operations
pub struct CuFileHandle {
    handle: sys::CUfileHandle_t,
    _file: File, // Keep the file alive
}

impl CuFileHandle {
    /// Register a file for CuFile operations
    ///
    /// # Arguments
    /// * `file` - The file to register for GPU-direct storage operations
    pub fn register(file: File) -> CuFileResult<Self> {
        let fd = file.as_raw_fd();
        let mut handle = ptr::null_mut();
        let mut descr = sys::CUfileDescr_t {
            type_: sys::CUfileFileHandleType::CU_FILE_HANDLE_TYPE_OPAQUE_FD,
            handle: sys::CUfileDescrHandle { fd: fd as c_int },
            fs_ops: ptr::null_mut(),
        };
        unsafe {
            check_cufile_error(sys::cuFileHandleRegister(&mut handle, &mut descr))?;
        }

        Ok(CuFileHandle {
            handle,
            _file: file,
        })
    }

    /// Get the raw CuFile handle
    pub(crate) fn raw_handle(&self) -> sys::CUfileHandle_t {
        self.handle
    }

    /// Read data from the file directly to GPU memory
    ///
    /// # Arguments
    /// * `dest_base` - Base address of buffer in device memory or host memory. For registered buffers, dest_base must remain set to the base address used in the cuFileBufRegister call.
    /// * `size` - Number of bytes to read.
    /// * `file_offset` - Offset in the file to start reading from.
    /// * `dest_offset` - Offset relative to the dest_base pointer to read into. This parameter should be used only with registered buffers.
    ///
    /// # Returns
    /// Size of bytes that were successfully read.
    pub fn read(
        &self,
        dest_base: *mut c_void,
        size: usize,
        file_offset: i64,
        dest_offset: i64,
    ) -> CuFileResult<isize> {
        unsafe {
            let ret = sys::cuFileRead(self.handle, dest_base, size, file_offset, dest_offset);
            if ret < 0 {
                //  TODO (Serapheim):
                // -1 on an error, so errno is set to indicate filesystem errors.
                // All other errors return a negative integer value of the CUfileOpError enum value.
                check_cufile_error((-ret).try_into().unwrap())?;
            }
            Ok(ret)
        }
    }

    /// Write data from GPU memory directly to the file
    ///
    /// # Arguments
    /// * `dest_base` - Base address of buffer in device memory or host memory. For registered buffers, dest_base must remain set to the base address used in the cuFileBufRegister call.
    /// * `size` - Number of bytes to write
    /// * `file_offset` - Offset in the file to start writing to
    /// * `dest_offset` - Offset relative to the dest_base pointer to read from. This parameter should be used only with registered buffers.
    ///
    /// # Returns
    /// The number of bytes successfully written
    pub fn write(&self, dest_base: *const c_void, size: usize, file_offset: i64, dest_offset: i64) -> CuFileResult<isize> {
        unsafe {
            let ret = sys::cuFileWrite(
                self.handle,
                dest_base,
                size,
                file_offset,
                dest_offset,
            );
            if ret < 0 {
                check_cufile_error((-ret).try_into().unwrap())?;
            }
            Ok(ret)
        }
    }
}

impl Drop for CuFileHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuFileHandleDeregister(self.handle);
        }
    }
}

// Safety: CuFileHandle can be safely sent between threads
unsafe impl Send for CuFileHandle {}
// Safety: CuFileHandle can be safely shared between threads with proper synchronization
unsafe impl Sync for CuFileHandle {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use tempfile::tempdir;

    #[test]
    fn test_handle_creation() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_file.dat");

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&file_path)
            .unwrap();

        // This test might fail if NVIDIA drivers/hardware aren't available
        match CuFileHandle::register(file) {
            Ok(_handle) => {
                println!("Handle created successfully");
            }
            Err(e) => {
                println!(
                    "Handle creation failed (expected in test environment): {:?}",
                    e
                );
            }
        }
    }
}
