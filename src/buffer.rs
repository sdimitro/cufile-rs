use crate::{check_cufile_error, sys, CuFileResult};
use std::marker::PhantomData;
use std::os::raw::c_void;

/// Flags for buffer registration
pub struct BufferFlags;

impl BufferFlags {
    pub const NONE: i32 = 0;
    // Add other flags as needed based on CuFile documentation
}

/// A registered GPU buffer for CuFile operations
pub struct RegisteredBuffer<T> {
    ptr: *mut c_void,
    size: usize,
    _phantom: PhantomData<T>,
}

impl<T> RegisteredBuffer<T> {
    /// Register a GPU buffer for CuFile operations
    ///
    /// # Arguments
    /// * `dev_ptr` - Pointer to GPU memory
    /// * `size` - Size of the buffer in bytes
    /// * `flags` - Registration flags
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `dev_ptr` points to valid GPU memory
    /// - The memory remains valid for the lifetime of this RegisteredBuffer
    /// - `size` accurately represents the size of the allocated memory
    pub unsafe fn register(dev_ptr: *mut T, size: usize, flags: i32) -> CuFileResult<Self> {
        let ptr = dev_ptr as *mut c_void;

        check_cufile_error(sys::cuFileBufRegister(ptr, size, flags))?;

        Ok(RegisteredBuffer {
            ptr,
            size,
            _phantom: PhantomData,
        })
    }

    /// Get the raw pointer to the registered buffer
    pub fn as_ptr(&self) -> *mut T {
        self.ptr as *mut T
    }

    /// Get the raw void pointer to the registered buffer
    pub fn as_void_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Get the size of the registered buffer in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the size of the registered buffer in elements
    pub fn len(&self) -> usize {
        self.size / std::mem::size_of::<T>()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl<T> Drop for RegisteredBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuFileBufDeregister(self.ptr);
        }
    }
}

// Safety: RegisteredBuffer can be safely sent between threads
unsafe impl<T> Send for RegisteredBuffer<T> {}
// Safety: RegisteredBuffer can be safely shared between threads with proper synchronization
unsafe impl<T> Sync for RegisteredBuffer<T> {}

/// Convenience functions for common buffer operations
impl RegisteredBuffer<u8> {
    /// Register a byte buffer
    ///
    /// # Safety
    /// See [`RegisteredBuffer::register`] for safety requirements
    pub unsafe fn register_bytes(dev_ptr: *mut u8, size: usize) -> CuFileResult<Self> {
        Self::register(dev_ptr, size, BufferFlags::NONE)
    }
}

impl<T> RegisteredBuffer<T>
where
    T: Copy,
{
    /// Register a typed buffer
    ///
    /// # Arguments
    /// * `dev_ptr` - Pointer to GPU memory
    /// * `count` - Number of elements of type T
    ///
    /// # Safety
    /// See [`RegisteredBuffer::register`] for safety requirements
    pub unsafe fn register_typed(dev_ptr: *mut T, count: usize) -> CuFileResult<Self> {
        let size = count * std::mem::size_of::<T>();
        Self::register(dev_ptr, size, BufferFlags::NONE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_size_calculations() {
        // Test size calculations without actually registering buffers
        let size_bytes = 1024;
        let count_u32 = 256; // 256 * 4 = 1024 bytes

        assert_eq!(count_u32 * std::mem::size_of::<u32>(), size_bytes);

        // Test with different types
        let count_f64 = 128; // 128 * 8 = 1024 bytes
        assert_eq!(count_f64 * std::mem::size_of::<f64>(), size_bytes);
    }

    #[test]
    fn test_buffer_flags() {
        assert_eq!(BufferFlags::NONE, 0);
    }
}
