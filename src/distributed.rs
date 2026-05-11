use crate::Exception;
use crate::ffi::ffi;
use crate::program::Program;

pub struct MeshDevice {
    pub(crate) inner: cxx::UniquePtr<ffi::MeshDeviceHandle>,
}

pub struct MeshWorkload {
    pub(crate) inner: cxx::UniquePtr<ffi::MeshWorkloadHandle>,
}

impl MeshDevice {
    pub fn create_unit_mesh(device_id: i32) -> Result<Self, Exception> {
        let inner = ffi::create_unit_mesh(device_id)?;
        Ok(Self { inner })
    }

    pub fn close(&mut self) -> Result<bool, Exception> {
        self.inner.pin_mut().close()
    }

    pub fn is_open(&self) -> bool {
        self.inner
            .as_ref()
            .map(ffi::MeshDeviceHandle::is_open)
            .unwrap_or(false)
    }

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
}

impl MeshWorkload {
    pub fn create() -> Self {
        Self {
            inner: ffi::create_mesh_workload(),
        }
    }

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

    pub fn program_count(&self) -> usize {
        self.inner
            .as_ref()
            .expect("mesh workload handle should exist")
            .program_count()
    }
}
