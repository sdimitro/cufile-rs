use std::ffi::CStr;
use thiserror::Error;

pub use cufile_sys as sys;

mod buffer;
mod driver;
mod handle;

pub use buffer::*;
pub use driver::*;
pub use handle::*;

/// CuFile error types
#[derive(Error, Debug, Clone, PartialEq)]
pub enum CuFileError {
    #[error("Success")]
    Success,
    #[error("Invalid value provided")]
    InvalidValue,
    #[error("I/O error occurred")]
    IoError,
    #[error("Permission denied")]
    PermissionDenied,
    #[error("Unknown error: {0}")]
    Unknown(i32),
    // TODO(serapheim): Add more
}

impl From<i32> for CuFileError {
    fn from(code: i32) -> Self {
        match code {
            sys::CU_FILE_SUCCESS => CuFileError::Success,
            sys::CU_FILE_INVALID_VALUE => CuFileError::InvalidValue,
            sys::CU_FILE_PERMISSION_DENIED => CuFileError::PermissionDenied,
            // TODO(serapheim): Add more
            code => CuFileError::Unknown(code),
        }
    }
}

/// Result type for CuFile operations
pub type CuFileResult<T> = Result<T, CuFileError>;

/// Convert a CuFile error code to a Result
pub fn check_cufile_error(code: sys::CUfileError_t) -> CuFileResult<()> {
    match code {
        sys::CU_FILE_SUCCESS => Ok(()),
        code => Err(CuFileError::from(code)),
    }
}

/// Get the error string for a CuFile error code
pub fn get_error_string(error: sys::CUfileError_t) -> String {
    unsafe {
        let ptr = sys::cuFileGetErrorString(error);
        if ptr.is_null() {
            format!("Unknown error: {}", error)
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_conversion() {
        assert_eq!(
            CuFileError::from(sys::CU_FILE_SUCCESS),
            CuFileError::Success
        );
        assert_eq!(
            CuFileError::from(sys::CU_FILE_INVALID_VALUE),
            CuFileError::InvalidValue
        );
    }

    #[test]
    fn test_check_cufile_error() {
        assert!(check_cufile_error(sys::CU_FILE_SUCCESS).is_ok());
        assert!(check_cufile_error(sys::CU_FILE_INVALID_VALUE).is_err());
    }
}
