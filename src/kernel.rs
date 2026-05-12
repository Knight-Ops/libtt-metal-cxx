use crate::Exception;
use crate::ffi::ffi;
use crate::program::Program;

pub type KernelId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalCore {
    pub x: u32,
    pub y: u32,
}

impl LogicalCore {
    pub const fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataMovementProcessor {
    Riscv0,
    Riscv1,
    Riscv2,
    Riscv3,
    Riscv4,
    Riscv5,
    Riscv6,
    Riscv7,
}

impl std::fmt::Display for DataMovementProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Riscv0 => write!(f, "Riscv0"),
            Self::Riscv1 => write!(f, "Riscv1"),
            Self::Riscv2 => write!(f, "Riscv2"),
            Self::Riscv3 => write!(f, "Riscv3"),
            Self::Riscv4 => write!(f, "Riscv4"),
            Self::Riscv5 => write!(f, "Riscv5"),
            Self::Riscv6 => write!(f, "Riscv6"),
            Self::Riscv7 => write!(f, "Riscv7"),
        }
    }
}

impl DataMovementProcessor {
    #[must_use]
    const fn as_ffi(self) -> u8 {
        match self {
            Self::Riscv0 => 0,
            Self::Riscv1 => 1,
            Self::Riscv2 => 2,
            Self::Riscv3 => 3,
            Self::Riscv4 => 4,
            Self::Riscv5 => 5,
            Self::Riscv6 => 6,
            Self::Riscv7 => 7,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Noc {
    Riscv0Default,
    Riscv1Default,
    Noc0,
    Noc1,
}

impl std::fmt::Display for Noc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Riscv0Default => write!(f, "Riscv0Default"),
            Self::Riscv1Default => write!(f, "Riscv1Default"),
            Self::Noc0 => write!(f, "Noc0"),
            Self::Noc1 => write!(f, "Noc1"),
        }
    }
}

impl Noc {
    #[must_use]
    const fn as_ffi(self) -> u8 {
        match self {
            Self::Riscv0Default => 0,
            Self::Riscv1Default => 1,
            Self::Noc0 => 2,
            Self::Noc1 => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NocMode {
    Dedicated,
    Dynamic,
}

impl std::fmt::Display for NocMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dedicated => write!(f, "Dedicated"),
            Self::Dynamic => write!(f, "Dynamic"),
        }
    }
}

impl NocMode {
    #[must_use]
    const fn as_ffi(self) -> u8 {
        match self {
            Self::Dedicated => 0,
            Self::Dynamic => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathFidelity {
    LoFi,
    HiFi2,
    HiFi3,
    HiFi4,
}

impl std::fmt::Display for MathFidelity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoFi => write!(f, "LoFi"),
            Self::HiFi2 => write!(f, "HiFi2"),
            Self::HiFi3 => write!(f, "HiFi3"),
            Self::HiFi4 => write!(f, "HiFi4"),
        }
    }
}

impl MathFidelity {
    #[must_use]
    const fn as_ffi(self) -> u8 {
        match self {
            Self::LoFi => 0,
            Self::HiFi2 => 2,
            Self::HiFi3 => 3,
            Self::HiFi4 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnpackToDestMode {
    UnpackToDestFp32,
    Default,
}

impl std::fmt::Display for UnpackToDestMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnpackToDestFp32 => write!(f, "UnpackToDestFp32"),
            Self::Default => write!(f, "Default"),
        }
    }
}

impl UnpackToDestMode {
    #[must_use]
    const fn as_ffi(self) -> u8 {
        match self {
            Self::UnpackToDestFp32 => 0,
            Self::Default => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelBuildOptLevel {
    O1,
    O2,
    O3,
    O0,
    Os,
    Ofast,
    Oz,
}

impl std::fmt::Display for KernelBuildOptLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::O1 => write!(f, "O1"),
            Self::O2 => write!(f, "O2"),
            Self::O3 => write!(f, "O3"),
            Self::O0 => write!(f, "O0"),
            Self::Os => write!(f, "Os"),
            Self::Ofast => write!(f, "Ofast"),
            Self::Oz => write!(f, "Oz"),
        }
    }
}

impl KernelBuildOptLevel {
    #[must_use]
    const fn as_ffi(self) -> u8 {
        match self {
            Self::O1 => 0,
            Self::O2 => 1,
            Self::O3 => 2,
            Self::O0 => 3,
            Self::Os => 4,
            Self::Ofast => 5,
            Self::Oz => 6,
        }
    }
}

pub struct ComputeKernelConfig {
    inner: cxx::UniquePtr<ffi::ComputeKernelConfigHandle>,
}

impl std::fmt::Debug for ComputeKernelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputeKernelConfig")
            .finish_non_exhaustive()
    }
}

impl Default for ComputeKernelConfig {
    fn default() -> Self {
        Self {
            inner: ffi::create_compute_kernel_config(),
        }
    }
}

impl ComputeKernelConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_math_fidelity(&mut self, math_fidelity: MathFidelity) -> &mut Self {
        self.inner
            .pin_mut()
            .set_math_fidelity(math_fidelity.as_ffi());
        self
    }

    pub fn set_fp32_dest_acc_en(&mut self, enabled: bool) -> &mut Self {
        self.inner.pin_mut().set_fp32_dest_acc_en(enabled);
        self
    }

    pub fn set_dst_full_sync_en(&mut self, enabled: bool) -> &mut Self {
        self.inner.pin_mut().set_dst_full_sync_en(enabled);
        self
    }

    pub fn set_unpack_to_dest_modes_all(&mut self, mode: UnpackToDestMode) -> &mut Self {
        self.inner
            .pin_mut()
            .fill_unpack_to_dest_modes(mode.as_ffi());
        self
    }

    pub fn set_bfp8_pack_precise(&mut self, enabled: bool) -> &mut Self {
        self.inner.pin_mut().set_bfp8_pack_precise(enabled);
        self
    }

    pub fn set_math_approx_mode(&mut self, enabled: bool) -> &mut Self {
        self.inner.pin_mut().set_math_approx_mode(enabled);
        self
    }

    pub fn add_compile_arg(&mut self, arg: u32) -> &mut Self {
        self.inner.pin_mut().add_compile_arg(arg);
        self
    }

    pub fn add_compile_args<I>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = u32>,
    {
        for arg in args {
            self.add_compile_arg(arg);
        }
        self
    }

    pub fn add_define(&mut self, key: &str, value: &str) -> &mut Self {
        self.inner.pin_mut().add_define(key, value);
        self
    }

    pub fn add_named_compile_arg(&mut self, key: &str, value: u32) -> &mut Self {
        self.inner.pin_mut().add_named_compile_arg(key, value);
        self
    }

    pub fn set_opt_level(&mut self, opt_level: KernelBuildOptLevel) -> &mut Self {
        self.inner.pin_mut().set_opt_level(opt_level.as_ffi());
        self
    }
}

pub struct DataMovementKernelConfig {
    inner: cxx::UniquePtr<ffi::DataMovementKernelConfigHandle>,
}

impl std::fmt::Debug for DataMovementKernelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataMovementKernelConfig")
            .finish_non_exhaustive()
    }
}

impl Default for DataMovementKernelConfig {
    fn default() -> Self {
        Self {
            inner: ffi::create_data_movement_kernel_config(),
        }
    }
}

impl DataMovementKernelConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a reader-style config using TT-Metal's preferred read-side defaults.
    ///
    /// This requires TT-Metal context to be initialized, so call it after opening a device or mesh.
    pub fn reader() -> Result<Self, Exception> {
        Ok(Self {
            inner: ffi::create_reader_data_movement_kernel_config()?,
        })
    }

    /// Creates a writer-style config using TT-Metal's preferred write-side defaults.
    ///
    /// This requires TT-Metal context to be initialized, so call it after opening a device or mesh.
    pub fn writer() -> Result<Self, Exception> {
        Ok(Self {
            inner: ffi::create_writer_data_movement_kernel_config()?,
        })
    }

    pub fn set_processor(&mut self, processor: DataMovementProcessor) -> &mut Self {
        self.inner.pin_mut().set_processor(processor.as_ffi());
        self
    }

    pub fn set_noc(&mut self, noc: Noc) -> &mut Self {
        self.inner.pin_mut().set_noc(noc.as_ffi());
        self
    }

    pub fn set_noc_mode(&mut self, noc_mode: NocMode) -> &mut Self {
        self.inner.pin_mut().set_noc_mode(noc_mode.as_ffi());
        self
    }

    pub fn add_compile_arg(&mut self, arg: u32) -> &mut Self {
        self.inner.pin_mut().add_compile_arg(arg);
        self
    }

    pub fn add_compile_args<I>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = u32>,
    {
        for arg in args {
            self.add_compile_arg(arg);
        }
        self
    }

    pub fn add_define(&mut self, key: &str, value: &str) -> &mut Self {
        self.inner.pin_mut().add_define(key, value);
        self
    }

    pub fn add_named_compile_arg(&mut self, key: &str, value: u32) -> &mut Self {
        self.inner.pin_mut().add_named_compile_arg(key, value);
        self
    }

    pub fn set_opt_level(&mut self, opt_level: KernelBuildOptLevel) -> &mut Self {
        self.inner.pin_mut().set_opt_level(opt_level.as_ffi());
        self
    }
}

impl Program {
    pub fn create_compute_kernel_from_string(
        &mut self,
        kernel_src_code: &str,
        core: LogicalCore,
    ) -> Result<KernelId, Exception> {
        self.inner
            .pin_mut()
            .create_compute_kernel_from_string(kernel_src_code, core.x, core.y)
    }

    pub fn create_compute_kernel(
        &mut self,
        file_name: &str,
        core: LogicalCore,
    ) -> Result<KernelId, Exception> {
        self.inner
            .pin_mut()
            .create_compute_kernel(file_name, core.x, core.y)
    }

    pub fn create_compute_kernel_from_string_with_config(
        &mut self,
        kernel_src_code: &str,
        core: LogicalCore,
        config: &ComputeKernelConfig,
    ) -> Result<KernelId, Exception> {
        self.inner
            .pin_mut()
            .create_compute_kernel_from_string_with_config(
                kernel_src_code,
                core.x,
                core.y,
                config
                    .inner
                    .as_ref()
                    .expect("compute kernel config handle should exist"),
            )
    }

    pub fn create_compute_kernel_with_config(
        &mut self,
        file_name: &str,
        core: LogicalCore,
        config: &ComputeKernelConfig,
    ) -> Result<KernelId, Exception> {
        self.inner.pin_mut().create_compute_kernel_with_config(
            file_name,
            core.x,
            core.y,
            config
                .inner
                .as_ref()
                .expect("compute kernel config handle should exist"),
        )
    }

    pub fn create_data_movement_kernel(
        &mut self,
        file_name: &str,
        core: LogicalCore,
        processor: DataMovementProcessor,
        noc: Noc,
    ) -> Result<KernelId, Exception> {
        self.inner.pin_mut().create_data_movement_kernel(
            file_name,
            core.x,
            core.y,
            processor.as_ffi(),
            noc.as_ffi(),
        )
    }

    pub fn create_data_movement_kernel_from_string(
        &mut self,
        kernel_src_code: &str,
        core: LogicalCore,
        processor: DataMovementProcessor,
        noc: Noc,
    ) -> Result<KernelId, Exception> {
        self.inner
            .pin_mut()
            .create_data_movement_kernel_from_string(
                kernel_src_code,
                core.x,
                core.y,
                processor.as_ffi(),
                noc.as_ffi(),
            )
    }

    pub fn create_data_movement_kernel_with_config(
        &mut self,
        file_name: &str,
        core: LogicalCore,
        config: &DataMovementKernelConfig,
    ) -> Result<KernelId, Exception> {
        self.inner
            .pin_mut()
            .create_data_movement_kernel_with_config(
                file_name,
                core.x,
                core.y,
                config
                    .inner
                    .as_ref()
                    .expect("data movement kernel config handle should exist"),
            )
    }

    pub fn create_data_movement_kernel_from_string_with_config(
        &mut self,
        kernel_src_code: &str,
        core: LogicalCore,
        config: &DataMovementKernelConfig,
    ) -> Result<KernelId, Exception> {
        self.inner
            .pin_mut()
            .create_data_movement_kernel_from_string_with_config(
                kernel_src_code,
                core.x,
                core.y,
                config
                    .inner
                    .as_ref()
                    .expect("data movement kernel config handle should exist"),
            )
    }
}
