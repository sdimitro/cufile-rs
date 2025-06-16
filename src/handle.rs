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

    /// Read data from the file directly into a buffer
    ///
    /// # Arguments
    /// * `buffer` - Mutable slice to read data into
    /// * `file_offset` - Offset in the file to start reading from
    /// * `dest_offset` - Offset relative to the buffer pointer to read into. This parameter should be used only with registered buffers.
    ///
    /// # Returns
    /// Number of bytes that were successfully read
    pub fn read(
        &self,
        buffer: &mut [u8],
        file_offset: i64,
        dest_offset: i64,
    ) -> CuFileResult<usize> {
        unsafe {
            let ret = self.read_raw(
                buffer.as_mut_ptr() as *mut c_void,
                buffer.len(),
                file_offset,
                dest_offset,
            )?;
            Ok(ret as usize)
        }
    }

    /// Read data from the file directly to GPU memory (low-level interface)
    ///
    /// This method provides direct access to CuFile's raw API for advanced users who need
    /// to work with GPU device pointers or require fine-grained control over memory layout.
    ///
    /// # Arguments
    /// * `dest_base` - Base address of buffer in device memory or host memory. For registered buffers, dest_base must remain set to the base address used in the cuFileBufRegister call.
    /// * `size` - Number of bytes to read.
    /// * `file_offset` - Offset in the file to start reading from.
    /// * `dest_offset` - Offset relative to the dest_base pointer to read into. This parameter should be used only with registered buffers.
    ///
    /// # Returns
    /// Size of bytes that were successfully read.
    ///
    /// # Safety
    /// The caller must ensure that `dest_base` points to a valid memory region of at least `size + dest_offset` bytes.
    pub unsafe fn read_raw(
        &self,
        dest_base: *mut c_void,
        size: usize,
        file_offset: i64,
        dest_offset: i64,
    ) -> CuFileResult<isize> {
        unsafe {
            let ret = sys::cuFileRead(self.handle, dest_base, size, file_offset, dest_offset);
            if ret < 0 {
                if ret == -1 {
                    // -1 indicates a filesystem error, so errno is set
                    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                    check_cufile_error(errno)?;
                } else {
                    // All other negative values are CUfileOpError enum values
                    panic!("Unexpected CuFile error code: {}", ret);
                }
            }
            Ok(ret)
        }
    }

    /// Write data from a buffer directly to the file
    ///
    /// # Arguments
    /// * `buffer` - Slice containing data to write
    /// * `file_offset` - Offset in the file to start writing to
    /// * `dest_offset` - Offset relative to the buffer pointer to read from. This parameter should be used only with registered buffers.
    ///
    /// # Returns
    /// Number of bytes that were successfully written
    pub fn write(&self, buffer: &[u8], file_offset: i64, dest_offset: i64) -> CuFileResult<usize> {
        unsafe {
            let ret = self.write_raw(
                buffer.as_ptr() as *const c_void,
                buffer.len(),
                file_offset,
                dest_offset,
            )?;
            Ok(ret as usize)
        }
    }

    /// Write data from GPU memory directly to the file (low-level interface)
    ///
    /// This method provides direct access to CuFile's raw API for advanced users who need
    /// to work with GPU device pointers or require fine-grained control over memory layout.
    ///
    /// # Arguments
    /// * `dest_base` - Base address of buffer in device memory or host memory. For registered buffers, dest_base must remain set to the base address used in the cuFileBufRegister call.
    /// * `size` - Number of bytes to write
    /// * `file_offset` - Offset in the file to start writing to
    /// * `dest_offset` - Offset relative to the dest_base pointer to read from. This parameter should be used only with registered buffers.
    ///
    /// # Returns
    /// The number of bytes successfully written
    ///
    /// # Safety
    /// The caller must ensure that `dest_base` points to a valid memory region of at least `size + dest_offset` bytes.
    pub unsafe fn write_raw(
        &self,
        dest_base: *const c_void,
        size: usize,
        file_offset: i64,
        dest_offset: i64,
    ) -> CuFileResult<isize> {
        unsafe {
            let ret = sys::cuFileWrite(self.handle, dest_base, size, file_offset, dest_offset);
            if ret < 0 {
                if ret == -1 {
                    // -1 indicates a filesystem error, so errno is set
                    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                    check_cufile_error(errno)?;
                } else {
                    // All other negative values are CUfileOpError enum values
                    panic!("Unexpected CuFile error code: {}", ret);
                }
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

// These tests might fail if NVIDIA drivers/hardware aren't available
#[cfg(test)]
mod tests {
    use crate::CuFileError;

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

        CuFileHandle::register(file).unwrap();
    }

    #[test]
    fn test_handle_already_registered() {
        use std::mem;
        use std::os::unix::io::{FromRawFd, IntoRawFd};

        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_file.dat");

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&file_path)
            .unwrap();

        // Get the raw FD and create two File handles from the same FD
        let raw_fd = file.into_raw_fd();
        let file1 = unsafe { File::from_raw_fd(raw_fd) };
        let file2 = unsafe { File::from_raw_fd(raw_fd) };

        let hdl1 = CuFileHandle::register(file1).unwrap();
        match CuFileHandle::register(file2) {
            Ok(_handle) => {
                assert!(
                    false,
                    "Handle created successfully even though it should have failed"
                );
            }
            Err(e) => {
                assert_eq!(
                    e,
                    CuFileError::HandleAlreadyRegistered,
                    "Handle creation failed (expected): {:?}",
                    e
                );
            }
        }

        // Prevent hdl1 from being dropped to avoid double-close
        // This leaks the handle but is acceptable in a test
        mem::forget(hdl1);
    }

    #[test]
    fn test_handle_invalid_file_type() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("/proc/self/fd/0");

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&file_path)
            .unwrap();

        match CuFileHandle::register(file) {
            Ok(_handle) => {
                assert!(
                    false,
                    "Handle created successfully even though it should have failed"
                );
            }
            Err(e) => {
                assert_eq!(
                    e,
                    CuFileError::InvalidFile,
                    "Handle creation failed (expected): {:?}",
                    e
                );
            }
        }
    }

    #[test]
    fn test_handle_read_permissions_error() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_file.dat");

        let file = OpenOptions::new()
            .create(true)
            .read(false)
            .write(true)
            .open(&file_path)
            .unwrap();

        let handle = CuFileHandle::register(file).unwrap();
        let mut buffer = [0u8; 10];
        let ret = handle.read(&mut buffer, 0, 0);
        // CuFile detects invalid file open flags before reaching filesystem level
        assert_eq!(ret, Err(CuFileError::InvalidFileOpenFlag));
    }
}
