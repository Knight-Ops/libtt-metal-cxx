use crate::Exception;
use crate::ffi::ffi;

/// A buffer allocated across the mesh of TT-Metal devices.
///
/// For a unit mesh (single device), this is a replicated buffer
/// that wraps a single device buffer.
pub struct MeshBuffer {
    pub(crate) inner: cxx::UniquePtr<ffi::MeshBufferHandle>,
}

impl std::fmt::Debug for MeshBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshBuffer")
            .field("address", &self.address())
            .field("size", &self.size())
            .field("is_allocated", &self.is_allocated())
            .finish()
    }
}

impl MeshBuffer {
    /// Create a replicated mesh buffer.
    ///
    /// For a unit mesh, each device gets a buffer of `size_bytes` bytes.
    /// `page_size` is the size of each page in bytes (typically `tile_size_bytes`).
    /// `buffer_type`: 0 = DRAM, 1 = L1, 2 = SYSTEM_MEMORY, 3 = L1_SMALL, 4 = TRACE.
    pub fn create_replicated(
        mesh_device: &crate::MeshDevice,
        size_bytes: u64,
        page_size: u64,
        buffer_type: u8,
    ) -> Result<Self, Exception> {
        let inner = ffi::create_replicated_mesh_buffer(
            mesh_device
                .inner
                .as_ref()
                .expect("mesh device handle should exist"),
            size_bytes,
            page_size,
            buffer_type,
        )?;
        Ok(Self { inner })
    }

    /// Returns the device address of the buffer (for use in runtime args).
    #[must_use]
    pub fn address(&self) -> u32 {
        self.inner
            .as_ref()
            .expect("mesh buffer handle should exist")
            .address()
    }

    /// Returns the size of the buffer in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.inner
            .as_ref()
            .expect("mesh buffer handle should exist")
            .size()
    }

    /// Returns whether the buffer is currently allocated.
    #[must_use]
    pub fn is_allocated(&self) -> bool {
        self.inner
            .as_ref()
            .map(ffi::MeshBufferHandle::is_allocated)
            .unwrap_or(false)
    }
}
