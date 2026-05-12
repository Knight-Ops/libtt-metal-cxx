use crate::ffi::ffi;

pub type ProgramId = u64;

pub struct Program {
    pub(crate) inner: cxx::UniquePtr<ffi::ProgramHandle>,
}

impl Program {
    pub fn new() -> Self {
        Self {
            inner: ffi::create_program(),
        }
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

impl Program {
    #[must_use]
    pub fn runtime_id(&self) -> Option<ProgramId> {
        self.inner.as_ref().map(ffi::ProgramHandle::runtime_id)
    }

    pub fn set_runtime_id(&mut self, runtime_id: ProgramId) {
        self.inner.pin_mut().set_runtime_id(runtime_id);
    }
}
