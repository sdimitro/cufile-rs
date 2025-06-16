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
    #[error("Handle already registered")]
    HandleAlreadyRegistered,
    #[error("FS or device I/O not supported")]
    IoNotSupported,
    #[error("Invalid file type")]
    InvalidFile,
    #[error("The nvidia-fs driver is not loaded")]
    DriverNotInitialized,
    #[error("Invalid property")]
    DriverInvalidProps,
    #[error("Property range error")]
    DriverUnsupportedLimit,
    #[error("nvidia-fs driver version mismatch")]
    DriverVersionMismatch,
    #[error("nvidia-fs driver version read error")]
    DriverVersionReadError,
    #[error("Driver shutdown in progress")]
    DriverClosing,
    #[error("GDS is not supported on the current platform")]
    PlatformNotSupported,
    #[error("GDS is not supported on the current GPU")]
    DeviceNotSupported,
    #[error("nvidia-fs driver ioctl error")]
    NvfsDriverError,
    #[error("CUDA Driver API error")]
    CudaDriverError,
    #[error("Invalid device pointer")]
    CudaPointerInvalid,
    #[error("Invalid pointer memory type")]
    CudaMemoryTypeInvalid,
    #[error("Pointer range exceeds the allocated address range")]
    CudaPointerRangeError,
    #[error("CUDA context mismatch")]
    CudaContextMismatch,
    #[error("Access beyond the maximum pinned memory size")]
    InvalidMappingSize,
    #[error("Access beyond the mapped size")]
    InvalidMappingRange,
    #[error("Unsupported file open flags")]
    InvalidFileOpenFlag,
    #[error("The fd direct IO is not set")]
    DioNotSet,
    #[error("Device pointer is already registered")]
    MemoryAlreadyRegistered,
    #[error("Device pointer lookup failure")]
    MemoryNotRegistered,
    #[error("The driver is already open")]
    DriverAlreadyOpen,
    #[error("The file descriptor is not registered")]
    HandleNotRegistered,
    #[error("The GPU device cannot be found")]
    DeviceNotFound,
    #[error("Internal error occurred. Refer to cufile.log for more details")]
    InternalError,
    #[error("Failed to obtain a new file descriptor")]
    GetNewFdFailed,
    #[error("NVFS driver initialization error")]
    NvfsSetupError,
    #[error("GDS is disabled by config on the current file")]
    IoDisabled,
    #[error("Failed to submit a batch operation")]
    BatchSubmitFailed,
    #[error("Failed to allocate pinned GPU memory")]
    GpuMemoryPinningFailed,
    #[error("Queue full for batch operation")]
    BatchFull,
    #[error("cuFile stream operation is not supported")]
    AsyncNotSupported,
}

impl From<i32> for CuFileError {
    fn from(code: i32) -> Self {
        match code {
            sys::CU_FILE_SUCCESS => CuFileError::Success,
            sys::CU_FILE_INVALID_VALUE => CuFileError::InvalidValue,
            sys::CU_FILE_PERMISSION_DENIED => CuFileError::PermissionDenied,
            sys::CU_FILE_HANDLE_ALREADY_REGISTERED => CuFileError::HandleAlreadyRegistered,
            sys::CU_FILE_IO_NOT_SUPPORTED => CuFileError::IoNotSupported,
            sys::CU_FILE_INVALID_FILE_TYPE => CuFileError::InvalidFile,
            sys::CU_FILE_DRIVER_NOT_INITIALIZED => CuFileError::DriverNotInitialized,
            sys::CU_FILE_DRIVER_INVALID_PROPS => CuFileError::DriverInvalidProps,
            sys::CU_FILE_DRIVER_UNSUPPORTED_LIMIT => CuFileError::DriverUnsupportedLimit,
            sys::CU_FILE_DRIVER_VERSION_MISMATCH => CuFileError::DriverVersionMismatch,
            sys::CU_FILE_DRIVER_VERSION_READ_ERROR => CuFileError::DriverVersionReadError,
            sys::CU_FILE_DRIVER_CLOSING => CuFileError::DriverClosing,
            sys::CU_FILE_PLATFORM_NOT_SUPPORTED => CuFileError::PlatformNotSupported,
            sys::CU_FILE_DEVICE_NOT_SUPPORTED => CuFileError::DeviceNotSupported,
            sys::CU_FILE_NVFS_DRIVER_ERROR => CuFileError::NvfsDriverError,
            sys::CU_FILE_CUDA_DRIVER_ERROR => CuFileError::CudaDriverError,
            sys::CU_FILE_CUDA_POINTER_INVALID => CuFileError::CudaPointerInvalid,
            sys::CU_FILE_CUDA_MEMORY_TYPE_INVALID => CuFileError::CudaMemoryTypeInvalid,
            sys::CU_FILE_CUDA_POINTER_RANGE_ERROR => CuFileError::CudaPointerRangeError,
            sys::CU_FILE_CUDA_CONTEXT_MISMATCH => CuFileError::CudaContextMismatch,
            sys::CU_FILE_INVALID_MAPPING_SIZE => CuFileError::InvalidMappingSize,
            sys::CU_FILE_INVALID_MAPPING_RANGE => CuFileError::InvalidMappingRange,
            sys::CU_FILE_INVALID_FILE_OPEN_FLAG => CuFileError::InvalidFileOpenFlag,
            sys::CU_FILE_DIO_NOT_SET => CuFileError::DioNotSet,
            sys::CU_FILE_MEMORY_ALREADY_REGISTERED => CuFileError::MemoryAlreadyRegistered,
            sys::CU_FILE_MEMORY_NOT_REGISTERED => CuFileError::MemoryNotRegistered,
            sys::CU_FILE_DRIVER_ALREADY_OPEN => CuFileError::DriverAlreadyOpen,
            sys::CU_FILE_HANDLE_NOT_REGISTERED => CuFileError::HandleNotRegistered,
            sys::CU_FILE_DEVICE_NOT_FOUND => CuFileError::DeviceNotFound,
            sys::CU_FILE_INTERNAL_ERROR => CuFileError::InternalError,
            sys::CU_FILE_GETNEWFD_FAILED => CuFileError::GetNewFdFailed,
            sys::CU_FILE_NVFS_SETUP_ERROR => CuFileError::NvfsSetupError,
            sys::CU_FILE_IO_DISABLED => CuFileError::IoDisabled,
            sys::CU_FILE_BATCH_SUBMIT_FAILED => CuFileError::BatchSubmitFailed,
            sys::CU_FILE_GPU_MEMORY_PINNING_FAILED => CuFileError::GpuMemoryPinningFailed,
            sys::CU_FILE_BATCH_FULL => CuFileError::BatchFull,
            sys::CU_FILE_ASYNC_NOT_SUPPORTED => CuFileError::AsyncNotSupported,
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
