use crate::Exception;
use crate::ffi::ffi;
use crate::mesh_buffer::MeshBuffer;
use crate::program::Program;

pub struct MeshDevice {
    pub(crate) inner: cxx::UniquePtr<ffi::MeshDeviceHandle>,
}

impl std::fmt::Debug for MeshDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshDevice")
            .field("device_id", &self.device_id())
            .field("is_open", &self.is_open())
            .finish()
    }
}

pub struct MeshWorkload {
    pub(crate) inner: cxx::UniquePtr<ffi::MeshWorkloadHandle>,
}

impl std::fmt::Debug for MeshWorkload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshWorkload")
            .field("program_count", &self.program_count())
            .finish()
    }
}

impl MeshDevice {
    pub fn create_unit_mesh(device_id: i32) -> Result<Self, Exception> {
        let inner = ffi::create_unit_mesh(device_id)?;
        Ok(Self { inner })
    }

    pub fn close(&mut self) -> Result<bool, Exception> {
        self.inner.pin_mut().close()
    }

    #[must_use]
    pub fn is_open(&self) -> bool {
        self.inner
            .as_ref()
            .map(ffi::MeshDeviceHandle::is_open)
            .unwrap_or(false)
    }

    #[must_use]
    pub fn device_id(&self) -> Option<i32> {
        self.inner.as_ref().map(ffi::MeshDeviceHandle::device_id)
    }

    pub fn num_devices(&self) -> Result<usize, Exception> {
        self.inner
            .as_ref()
            .expect("mesh device handle should exist")
            .num_devices()
    }

    pub fn num_rows(&self) -> Result<usize, Exception> {
        self.inner
            .as_ref()
            .expect("mesh device handle should exist")
            .num_rows()
    }

    pub fn num_cols(&self) -> Result<usize, Exception> {
        self.inner
            .as_ref()
            .expect("mesh device handle should exist")
            .num_cols()
    }

    pub fn enqueue_workload(
        &self,
        workload: &mut MeshWorkload,
        blocking: bool,
    ) -> Result<(), Exception> {
        self.inner
            .as_ref()
            .expect("mesh device handle should exist")
            .enqueue_workload(workload.inner.pin_mut(), blocking)
    }

    /// Write host data to an allocated mesh buffer (blocking).
    ///
    /// The data slice length must match the buffer size.
    pub fn write_mesh_buffer(&self, buffer: &MeshBuffer, data: &[u8]) -> Result<(), Exception> {
        self.inner
            .as_ref()
            .expect("mesh device handle should exist")
            .write_mesh_buffer(&buffer.inner, data)
    }

    /// Read device data from an allocated mesh buffer into host memory (blocking).
    ///
    /// The data slice length must match the buffer size.
    pub fn read_mesh_buffer(&self, buffer: &MeshBuffer, data: &mut [u8]) -> Result<(), Exception> {
        self.inner
            .as_ref()
            .expect("mesh device handle should exist")
            .read_mesh_buffer(&buffer.inner, data)
    }
}

impl MeshWorkload {
    pub fn new() -> Self {
        Self {
            inner: ffi::create_mesh_workload(),
        }
    }
}

impl Default for MeshWorkload {
    fn default() -> Self {
        Self::new()
    }
}

impl MeshWorkload {
    pub fn add_program_to_full_mesh(
        &mut self,
        mesh_device: &MeshDevice,
        program: Program,
    ) -> Result<(), Exception> {
        self.inner.pin_mut().add_program_to_full_mesh(
            mesh_device
                .inner
                .as_ref()
                .expect("mesh device handle should exist"),
            program.inner,
        )
    }

    #[must_use]
    pub fn program_count(&self) -> usize {
        self.inner
            .as_ref()
            .expect("mesh workload handle should exist")
            .program_count()
    }
}
