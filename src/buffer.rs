use crate::Exception;
use crate::device::Device;
use crate::ffi::ffi;
use crate::kernel::LogicalCore;
use crate::program::Program;

pub type CircularBufferId = usize;
pub type SemaphoreId = u32;

fn invalid_argument<T>(message: &str) -> Result<T, Exception> {
    match ffi::throw_invalid_argument(message) {
        Ok(()) => unreachable!("throw_invalid_argument should never return Ok"),
        Err(error) => Err(error),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubDeviceId(pub u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferType {
    Dram,
    L1,
    SystemMemory,
    L1Small,
    Trace,
}

impl BufferType {
    const fn as_ffi(self) -> u8 {
        match self {
            Self::Dram => 0,
            Self::L1 => 1,
            Self::SystemMemory => 2,
            Self::L1Small => 3,
            Self::Trace => 4,
        }
    }

    fn from_ffi(value: u8) -> Self {
        match value {
            0 => Self::Dram,
            1 => Self::L1,
            2 => Self::SystemMemory,
            3 => Self::L1Small,
            4 => Self::Trace,
            _ => panic!("unexpected BufferType value from C++: {value}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMemoryLayout {
    Interleaved,
    HeightSharded,
    WidthSharded,
    BlockSharded,
    NdSharded,
}

impl TensorMemoryLayout {
    const fn as_ffi(self) -> u8 {
        match self {
            Self::Interleaved => 0,
            Self::HeightSharded => 2,
            Self::WidthSharded => 3,
            Self::BlockSharded => 4,
            Self::NdSharded => 5,
        }
    }

    fn from_ffi(value: u8) -> Self {
        match value {
            0 => Self::Interleaved,
            2 => Self::HeightSharded,
            3 => Self::WidthSharded,
            4 => Self::BlockSharded,
            5 => Self::NdSharded,
            _ => panic!("unexpected TensorMemoryLayout value from C++: {value}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardOrientation {
    RowMajor,
    ColMajor,
}

impl ShardOrientation {
    const fn as_ffi(self) -> u8 {
        match self {
            Self::RowMajor => 0,
            Self::ColMajor => 1,
        }
    }

    fn from_ffi(value: u8) -> Self {
        match value {
            0 => Self::RowMajor,
            1 => Self::ColMajor,
            _ => panic!("unexpected ShardOrientation value from C++: {value}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    Float32,
    Float16,
    Bfp8,
    Bfp4,
    Bfp2,
    Float16B,
    Bfp8B,
    Bfp4B,
    Bfp2B,
    Lf8,
    Fp8E4M3,
    Int8,
    Tf32,
    UInt8,
    UInt16,
    Int16,
    Int32,
    UInt32,
    RawUInt8,
    RawUInt16,
    RawUInt32,
    Invalid,
}

impl DataFormat {
    const fn as_ffi(self) -> u8 {
        match self {
            Self::Float32 => 0,
            Self::Float16 => 1,
            Self::Bfp8 => 2,
            Self::Bfp4 => 3,
            Self::Bfp2 => 11,
            Self::Float16B => 5,
            Self::Bfp8B => 6,
            Self::Bfp4B => 7,
            Self::Bfp2B => 15,
            Self::Lf8 => 10,
            Self::Fp8E4M3 => 26,
            Self::Int8 => 14,
            Self::Tf32 => 4,
            Self::UInt8 => 30,
            Self::UInt16 => 9,
            Self::Int16 => 13,
            Self::Int32 => 8,
            Self::UInt32 => 24,
            Self::RawUInt8 => 240,
            Self::RawUInt16 => 241,
            Self::RawUInt32 => 242,
            Self::Invalid => 255,
        }
    }

    fn from_ffi(value: u8) -> Self {
        match value {
            0 => Self::Float32,
            1 => Self::Float16,
            2 => Self::Bfp8,
            3 => Self::Bfp4,
            11 => Self::Bfp2,
            5 => Self::Float16B,
            6 => Self::Bfp8B,
            7 => Self::Bfp4B,
            15 => Self::Bfp2B,
            10 => Self::Lf8,
            26 => Self::Fp8E4M3,
            14 => Self::Int8,
            4 => Self::Tf32,
            30 => Self::UInt8,
            9 => Self::UInt16,
            13 => Self::Int16,
            8 => Self::Int32,
            24 => Self::UInt32,
            240 => Self::RawUInt8,
            241 => Self::RawUInt16,
            242 => Self::RawUInt32,
            255 => Self::Invalid,
            _ => panic!("unexpected DataFormat value from C++: {value}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreRange {
    pub start: LogicalCore,
    pub end: LogicalCore,
}

impl CoreRange {
    pub const fn new(start: LogicalCore, end: LogicalCore) -> Self {
        Self { start, end }
    }

    pub const fn from_core(core: LogicalCore) -> Self {
        Self {
            start: core,
            end: core,
        }
    }

    fn as_ffi(self) -> ffi::CoreRangeRepr {
        ffi::CoreRangeRepr {
            start_x: self.start.x,
            start_y: self.start.y,
            end_x: self.end.x,
            end_y: self.end.y,
        }
    }

    fn from_ffi(repr: ffi::CoreRangeRepr) -> Self {
        Self {
            start: LogicalCore::new(repr.start_x, repr.start_y),
            end: LogicalCore::new(repr.end_x, repr.end_y),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CoreRangeSet {
    ranges: Vec<CoreRange>,
}

impl CoreRangeSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_core(core: LogicalCore) -> Self {
        Self::from_range(CoreRange::from_core(core))
    }

    pub fn from_range(range: CoreRange) -> Self {
        Self {
            ranges: vec![range],
        }
    }

    pub fn from_ranges<I>(ranges: I) -> Self
    where
        I: IntoIterator<Item = CoreRange>,
    {
        Self {
            ranges: ranges.into_iter().collect(),
        }
    }

    pub fn push(&mut self, range: CoreRange) {
        self.ranges.push(range);
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    pub fn ranges(&self) -> &[CoreRange] {
        &self.ranges
    }

    fn ffi_ranges(&self) -> Vec<ffi::CoreRangeRepr> {
        self.ranges.iter().copied().map(CoreRange::as_ffi).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardSpec {
    pub shape: [u32; 2],
    pub orientation: ShardOrientation,
}

impl ShardSpec {
    pub const fn new(shape: [u32; 2], orientation: ShardOrientation) -> Self {
        Self { shape, orientation }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardSpecBuffer {
    pub grid: CoreRangeSet,
    pub shard_spec: ShardSpec,
    pub page_shape: [u32; 2],
    pub tensor2d_shape_in_pages: [u32; 2],
}

impl ShardSpecBuffer {
    pub fn new(
        grid: CoreRangeSet,
        shape: [u32; 2],
        orientation: ShardOrientation,
        page_shape: [u32; 2],
        tensor2d_shape_in_pages: [u32; 2],
    ) -> Self {
        Self {
            grid,
            shard_spec: ShardSpec::new(shape, orientation),
            page_shape,
            tensor2d_shape_in_pages,
        }
    }

    fn from_ffi(
        metadata: ffi::ShardSpecBufferMetadataRepr,
        ranges: Vec<ffi::CoreRangeRepr>,
    ) -> Self {
        Self {
            grid: CoreRangeSet::from_ranges(ranges.into_iter().map(CoreRange::from_ffi)),
            shard_spec: ShardSpec::new(
                [metadata.shard_shape_height, metadata.shard_shape_width],
                ShardOrientation::from_ffi(metadata.shard_orientation),
            ),
            page_shape: [metadata.page_shape_height, metadata.page_shape_width],
            tensor2d_shape_in_pages: [
                metadata.tensor2d_shape_height,
                metadata.tensor2d_shape_width,
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileConfig {
    pub height: u32,
    pub width: u32,
    pub transpose_tile: bool,
}

impl TileConfig {
    pub const fn new(height: u32, width: u32, transpose_tile: bool) -> Self {
        Self {
            height,
            width,
            transpose_tile,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterleavedBufferConfig {
    pub size: u64,
    pub page_size: u64,
    pub buffer_type: BufferType,
}

impl InterleavedBufferConfig {
    pub const fn new(size: u64, page_size: u64, buffer_type: BufferType) -> Self {
        Self {
            size,
            page_size,
            buffer_type,
        }
    }

    fn as_ffi(self) -> ffi::InterleavedBufferConfigRepr {
        ffi::InterleavedBufferConfigRepr {
            size: self.size,
            page_size: self.page_size,
            buffer_type: self.buffer_type.as_ffi(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedBufferConfig {
    pub size: u64,
    pub page_size: u64,
    pub buffer_type: BufferType,
    pub buffer_layout: TensorMemoryLayout,
    pub shard_spec: ShardSpecBuffer,
}

impl ShardedBufferConfig {
    pub fn new(
        size: u64,
        page_size: u64,
        buffer_type: BufferType,
        buffer_layout: TensorMemoryLayout,
        shard_spec: ShardSpecBuffer,
    ) -> Self {
        Self {
            size,
            page_size,
            buffer_type,
            buffer_layout,
            shard_spec,
        }
    }

    fn as_ffi(&self) -> ffi::ShardedBufferConfigRepr {
        ffi::ShardedBufferConfigRepr {
            size: self.size,
            page_size: self.page_size,
            buffer_type: self.buffer_type.as_ffi(),
            buffer_layout: self.buffer_layout.as_ffi(),
            shard_orientation: self.shard_spec.shard_spec.orientation.as_ffi(),
            shard_shape_height: self.shard_spec.shard_spec.shape[0],
            shard_shape_width: self.shard_spec.shard_spec.shape[1],
            page_shape_height: self.shard_spec.page_shape[0],
            page_shape_width: self.shard_spec.page_shape[1],
            tensor2d_shape_height: self.shard_spec.tensor2d_shape_in_pages[0],
            tensor2d_shape_width: self.shard_spec.tensor2d_shape_in_pages[1],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BufferCreateOptions {
    pub address: Option<u64>,
    pub sub_device_id: Option<SubDeviceId>,
}

impl BufferCreateOptions {
    pub const fn new() -> Self {
        Self {
            address: None,
            sub_device_id: None,
        }
    }

    pub const fn with_address(mut self, address: u64) -> Self {
        self.address = Some(address);
        self
    }

    pub const fn with_sub_device_id(mut self, sub_device_id: SubDeviceId) -> Self {
        self.sub_device_id = Some(sub_device_id);
        self
    }

    fn as_ffi(self) -> Result<ffi::BufferCreateOptionsRepr, Exception> {
        if self.address.is_some() && self.sub_device_id.is_some() {
            return invalid_argument(
                "BufferCreateOptions cannot specify both a fixed address and a sub-device id",
            );
        }

        Ok(ffi::BufferCreateOptionsRepr {
            has_address: self.address.is_some(),
            address: self.address.unwrap_or_default(),
            has_sub_device_id: self.sub_device_id.is_some(),
            sub_device_id: self.sub_device_id.map(|id| id.0).unwrap_or_default(),
        })
    }
}

pub struct Buffer {
    pub(crate) inner: cxx::UniquePtr<ffi::BufferHandle>,
}

impl Buffer {
    pub fn create_interleaved(
        device: &Device,
        config: InterleavedBufferConfig,
        options: BufferCreateOptions,
    ) -> Result<Self, Exception> {
        let inner = ffi::create_interleaved_buffer(
            device.inner.as_ref().expect("device handle should exist"),
            &config.as_ffi(),
            &options.as_ffi()?,
        )?;
        Ok(Self { inner })
    }

    pub fn create_sharded(
        device: &Device,
        config: &ShardedBufferConfig,
        options: BufferCreateOptions,
    ) -> Result<Self, Exception> {
        let core_ranges = config.shard_spec.grid.ffi_ranges();
        let inner = ffi::create_sharded_buffer(
            device.inner.as_ref().expect("device handle should exist"),
            &config.as_ffi(),
            &core_ranges,
            &options.as_ffi()?,
        )?;
        Ok(Self { inner })
    }

    fn info(&self) -> ffi::BufferInfoRepr {
        self.inner
            .as_ref()
            .expect("buffer handle should exist")
            .info()
    }

    pub fn is_allocated(&self) -> bool {
        self.info().is_allocated
    }

    pub fn address(&self) -> u32 {
        self.info().address
    }

    pub fn size(&self) -> u64 {
        self.info().size
    }

    pub fn page_size(&self) -> u64 {
        self.info().page_size
    }

    pub fn buffer_type(&self) -> BufferType {
        BufferType::from_ffi(self.info().buffer_type)
    }

    pub fn buffer_layout(&self) -> TensorMemoryLayout {
        TensorMemoryLayout::from_ffi(self.info().buffer_layout)
    }

    pub fn sub_device_id(&self) -> Option<SubDeviceId> {
        let info = self.info();
        info.has_sub_device_id
            .then_some(SubDeviceId(info.sub_device_id))
    }

    pub fn shard_spec(&self) -> Result<Option<ShardSpecBuffer>, Exception> {
        let Some(handle) = self.inner.as_ref() else {
            return Ok(None);
        };
        if !handle.has_shard_spec() {
            return Ok(None);
        }

        let metadata = handle.shard_spec_metadata()?;
        let ranges = handle.shard_spec_core_ranges()?;
        Ok(Some(ShardSpecBuffer::from_ffi(metadata, ranges)))
    }

    pub fn deallocate(&mut self) -> Result<bool, Exception> {
        self.inner.pin_mut().deallocate()
    }
}

pub struct CircularBufferConfig {
    inner: cxx::UniquePtr<ffi::CircularBufferConfigHandle>,
}

impl CircularBufferConfig {
    pub fn new(total_size: u32) -> Self {
        Self {
            inner: ffi::create_circular_buffer_config(total_size),
        }
    }

    pub fn set_total_size(&mut self, total_size: u32) -> &mut Self {
        self.inner.pin_mut().set_total_size(total_size);
        self
    }

    pub fn set_address_offset(&mut self, offset: u32) -> &mut Self {
        self.inner.pin_mut().set_address_offset(offset);
        self
    }

    pub fn set_globally_allocated_address(
        &mut self,
        buffer: &Buffer,
    ) -> Result<&mut Self, Exception> {
        self.inner.pin_mut().set_globally_allocated_address(
            buffer.inner.as_ref().expect("buffer handle should exist"),
        )?;
        Ok(self)
    }

    pub fn set_globally_allocated_address_and_total_size(
        &mut self,
        buffer: &Buffer,
        total_size: u32,
    ) -> Result<&mut Self, Exception> {
        self.inner
            .pin_mut()
            .set_globally_allocated_address_and_total_size(
                buffer.inner.as_ref().expect("buffer handle should exist"),
                total_size,
            )?;
        Ok(self)
    }

    pub fn index(&mut self, buffer_index: u8) -> CircularBufferIndex<'_> {
        CircularBufferIndex {
            config: self,
            buffer_index,
            remote: false,
        }
    }

    pub fn remote_index(&mut self, buffer_index: u8) -> CircularBufferIndex<'_> {
        CircularBufferIndex {
            config: self,
            buffer_index,
            remote: true,
        }
    }
}

pub struct CircularBufferIndex<'a> {
    config: &'a mut CircularBufferConfig,
    buffer_index: u8,
    remote: bool,
}

impl CircularBufferIndex<'_> {
    pub fn set_data_format(&mut self, data_format: DataFormat) -> &mut Self {
        self.config.inner.pin_mut().set_index_data_format(
            self.buffer_index,
            self.remote,
            data_format.as_ffi(),
        );
        self
    }

    pub fn set_total_size(&mut self, total_size: u32) -> &mut Self {
        self.config.inner.pin_mut().set_index_total_size(
            self.buffer_index,
            self.remote,
            total_size,
        );
        self
    }

    pub fn set_page_size(&mut self, page_size: u32) -> &mut Self {
        self.config
            .inner
            .pin_mut()
            .set_index_page_size(self.buffer_index, self.remote, page_size);
        self
    }

    pub fn set_tile(&mut self, tile: TileConfig) -> Result<&mut Self, Exception> {
        self.config.inner.pin_mut().set_index_tile(
            self.buffer_index,
            self.remote,
            tile.height,
            tile.width,
            tile.transpose_tile,
        )?;
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CircularBufferIndexConfig {
    pub buffer_index: u8,
    pub is_remote: bool,
    pub data_format: Option<DataFormat>,
    pub page_size: Option<u32>,
    pub tile: Option<TileConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CircularBufferConfigSnapshot {
    pub total_size: u32,
    pub globally_allocated_address: Option<u32>,
    pub dynamic_cb: bool,
    pub max_size: u32,
    pub buffer_size: u32,
    pub address_offset: u32,
    pub indices: Vec<CircularBufferIndexConfig>,
}

impl Program {
    pub fn assign_global_buffer(&mut self, buffer: &Buffer) -> Result<(), Exception> {
        self.inner
            .pin_mut()
            .assign_global_buffer(buffer.inner.as_ref().expect("buffer handle should exist"))
    }

    pub fn create_circular_buffer(
        &mut self,
        core_ranges: &CoreRangeSet,
        config: &CircularBufferConfig,
    ) -> Result<CircularBufferId, Exception> {
        if core_ranges.is_empty() {
            return invalid_argument("create_circular_buffer requires at least one core range");
        }

        self.inner.pin_mut().create_circular_buffer(
            &core_ranges.ffi_ranges(),
            config
                .inner
                .as_ref()
                .expect("circular buffer config handle should exist"),
        )
    }

    pub fn circular_buffer_config(
        &self,
        circular_buffer_id: CircularBufferId,
    ) -> Result<CircularBufferConfigSnapshot, Exception> {
        let metadata = self
            .inner
            .get_circular_buffer_metadata(circular_buffer_id)?;
        let indices = self
            .inner
            .get_circular_buffer_indices(circular_buffer_id)?
            .into_iter()
            .map(|index| CircularBufferIndexConfig {
                buffer_index: index.buffer_index,
                is_remote: index.is_remote,
                data_format: index
                    .has_data_format
                    .then(|| DataFormat::from_ffi(index.data_format)),
                page_size: index.has_page_size.then_some(index.page_size),
                tile: index.has_tile.then_some(TileConfig::new(
                    index.tile_height,
                    index.tile_width,
                    index.tile_transpose,
                )),
            })
            .collect();

        Ok(CircularBufferConfigSnapshot {
            total_size: metadata.total_size,
            globally_allocated_address: metadata
                .has_globally_allocated_address
                .then_some(metadata.globally_allocated_address),
            dynamic_cb: metadata.dynamic_cb,
            max_size: metadata.max_size,
            buffer_size: metadata.buffer_size,
            address_offset: metadata.address_offset,
            indices,
        })
    }

    pub fn update_circular_buffer_total_size(
        &mut self,
        circular_buffer_id: CircularBufferId,
        total_size: u32,
    ) -> Result<(), Exception> {
        self.inner
            .pin_mut()
            .update_circular_buffer_total_size(circular_buffer_id, total_size)
    }

    pub fn update_circular_buffer_page_size(
        &mut self,
        circular_buffer_id: CircularBufferId,
        buffer_index: u8,
        page_size: u32,
    ) -> Result<(), Exception> {
        self.inner.pin_mut().update_circular_buffer_page_size(
            circular_buffer_id,
            buffer_index,
            page_size,
        )
    }

    pub fn update_dynamic_circular_buffer_address(
        &mut self,
        circular_buffer_id: CircularBufferId,
        buffer: &Buffer,
    ) -> Result<(), Exception> {
        self.inner.pin_mut().update_dynamic_circular_buffer_address(
            circular_buffer_id,
            buffer.inner.as_ref().expect("buffer handle should exist"),
        )
    }

    pub fn update_dynamic_circular_buffer_address_with_offset(
        &mut self,
        circular_buffer_id: CircularBufferId,
        buffer: &Buffer,
        address_offset: u32,
    ) -> Result<(), Exception> {
        self.inner
            .pin_mut()
            .update_dynamic_circular_buffer_address_with_offset(
                circular_buffer_id,
                buffer.inner.as_ref().expect("buffer handle should exist"),
                address_offset,
            )
    }

    pub fn update_dynamic_circular_buffer_address_and_total_size(
        &mut self,
        circular_buffer_id: CircularBufferId,
        buffer: &Buffer,
        total_size: u32,
    ) -> Result<(), Exception> {
        self.inner
            .pin_mut()
            .update_dynamic_circular_buffer_address_and_total_size(
                circular_buffer_id,
                buffer.inner.as_ref().expect("buffer handle should exist"),
                total_size,
            )
    }

    pub fn create_semaphore(
        &mut self,
        core_ranges: &CoreRangeSet,
        initial_value: u32,
    ) -> Result<SemaphoreId, Exception> {
        if core_ranges.is_empty() {
            return invalid_argument("create_semaphore requires at least one core range");
        }

        self.inner
            .pin_mut()
            .create_semaphore(&core_ranges.ffi_ranges(), initial_value)
    }
}

#[cfg(test)]
mod tests {
    use super::{BufferCreateOptions, SubDeviceId};

    #[test]
    fn buffer_create_options_reject_address_and_sub_device_together() {
        let options = BufferCreateOptions::new()
            .with_address(0x1000)
            .with_sub_device_id(SubDeviceId(3));
        let error = match options.as_ffi() {
            Ok(_) => panic!("conflicting buffer create options should be rejected"),
            Err(error) => error,
        };
        assert!(
            error.what().contains("cannot specify both"),
            "unexpected error message: {}",
            error.what()
        );
    }
}
