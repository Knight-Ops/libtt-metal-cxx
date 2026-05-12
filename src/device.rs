use crate::Exception;
use crate::ffi::ffi;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCounts {
    pub available: usize,
    pub pcie: usize,
}

pub struct Device {
    pub(crate) inner: cxx::UniquePtr<ffi::DeviceHandle>,
}

impl Device {
    pub fn create(device_id: i32) -> Result<Self, Exception> {
        let inner = ffi::create_device(device_id)?;
        Ok(Self { inner })
    }

    pub fn close(&mut self) -> Result<bool, Exception> {
        self.inner.pin_mut().close()
    }

    #[must_use]
    pub fn is_open(&self) -> bool {
        self.inner
            .as_ref()
            .map(ffi::DeviceHandle::is_open)
            .unwrap_or(false)
    }

    #[must_use]
    pub fn device_id(&self) -> Option<i32> {
        self.inner.as_ref().map(ffi::DeviceHandle::device_id)
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("device_id", &self.device_id())
            .field("is_open", &self.is_open())
            .finish()
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
