use crate::Exception;
use crate::kernel::{KernelId, LogicalCore};
use crate::program::Program;

impl Program {
    pub fn set_runtime_args(
        &mut self,
        kernel_id: KernelId,
        core: LogicalCore,
        runtime_args: &[u32],
    ) -> Result<(), Exception> {
        self.inner
            .pin_mut()
            .set_runtime_args(kernel_id, core.x, core.y, runtime_args)
    }

    pub fn runtime_args(
        &self,
        kernel_id: KernelId,
        core: LogicalCore,
    ) -> Result<Vec<u32>, Exception> {
        self.inner.get_runtime_args(kernel_id, core.x, core.y)
    }

    pub fn set_common_runtime_args(
        &mut self,
        kernel_id: KernelId,
        runtime_args: &[u32],
    ) -> Result<(), Exception> {
        self.inner
            .pin_mut()
            .set_common_runtime_args(kernel_id, runtime_args)
    }

    pub fn common_runtime_args(&self, kernel_id: KernelId) -> Result<Vec<u32>, Exception> {
        self.inner.get_common_runtime_args(kernel_id)
    }
}
