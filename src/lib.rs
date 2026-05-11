#[cxx::bridge(namespace = "tt_metal_cxx")]
mod ffi {
    unsafe extern "C++" {
        include!("libtt-metal-cxx/include/tt_metal_cxx.hpp");

        type DeviceHandle;
        type ProgramHandle;

        fn create_device(device_id: i32) -> Result<UniquePtr<DeviceHandle>>;
        fn create_program() -> UniquePtr<ProgramHandle>;
        fn close(self: Pin<&mut DeviceHandle>) -> Result<bool>;
        fn is_open(self: &DeviceHandle) -> bool;
        fn device_id(self: &DeviceHandle) -> i32;
        fn runtime_id(self: &ProgramHandle) -> u64;
        fn set_runtime_id(self: Pin<&mut ProgramHandle>, runtime_id: u64);

        fn get_num_available_devices() -> Result<usize>;
        fn get_num_pcie_devices() -> Result<usize>;
    }
}

pub use cxx::Exception;
pub type ProgramId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCounts {
    pub available: usize,
    pub pcie: usize,
}

pub struct Device {
    inner: cxx::UniquePtr<ffi::DeviceHandle>,
}

pub struct Program {
    inner: cxx::UniquePtr<ffi::ProgramHandle>,
}

impl Device {
    pub fn create(device_id: i32) -> Result<Self, Exception> {
        let inner = ffi::create_device(device_id)?;
        Ok(Self { inner })
    }

    pub fn close(&mut self) -> Result<bool, Exception> {
        self.inner.pin_mut().close()
    }

    pub fn is_open(&self) -> bool {
        self.inner
            .as_ref()
            .map(ffi::DeviceHandle::is_open)
            .unwrap_or(false)
    }

    pub fn device_id(&self) -> Option<i32> {
        self.inner.as_ref().map(ffi::DeviceHandle::device_id)
    }
}

impl Program {
    pub fn create() -> Self {
        Self {
            inner: ffi::create_program(),
        }
    }

    pub fn runtime_id(&self) -> Option<ProgramId> {
        self.inner.as_ref().map(ffi::ProgramHandle::runtime_id)
    }

    pub fn set_runtime_id(&mut self, runtime_id: ProgramId) {
        self.inner.pin_mut().set_runtime_id(runtime_id);
    }
}

pub fn query_devices() -> Result<DeviceCounts, Exception> {
    Ok(DeviceCounts {
        available: ffi::get_num_available_devices()?,
        pcie: ffi::get_num_pcie_devices()?,
    })
}

pub fn available_device_count() -> Result<usize, Exception> {
    Ok(query_devices()?.available)
}

pub fn pcie_device_count() -> Result<usize, Exception> {
    Ok(query_devices()?.pcie)
}
