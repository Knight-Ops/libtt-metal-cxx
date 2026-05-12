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

impl std::fmt::Display for BufferType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dram => write!(f, "DRAM"),
            Self::L1 => write!(f, "L1"),
            Self::SystemMemory => write!(f, "SystemMemory"),
            Self::L1Small => write!(f, "L1Small"),
            Self::Trace => write!(f, "Trace"),
        }
    }
}

impl BufferType {
    #[must_use]
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

impl std::fmt::Display for TensorMemoryLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Interleaved => write!(f, "Interleaved"),
            Self::HeightSharded => write!(f, "HeightSharded"),
            Self::WidthSharded => write!(f, "WidthSharded"),
            Self::BlockSharded => write!(f, "BlockSharded"),
            Self::NdSharded => write!(f, "NdSharded"),
        }
    }
}

impl TensorMemoryLayout {
    #[must_use]
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

impl std::fmt::Display for ShardOrientation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RowMajor => write!(f, "RowMajor"),
            Self::ColMajor => write!(f, "ColMajor"),
        }
    }
}

impl ShardOrientation {
    #[must_use]
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

impl std::fmt::Display for DataFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float32 => write!(f, "Float32"),
            Self::Float16 => write!(f, "Float16"),
            Self::Bfp8 => write!(f, "Bfp8"),
            Self::Bfp4 => write!(f, "Bfp4"),
            Self::Bfp2 => write!(f, "Bfp2"),
            Self::Float16B => write!(f, "Float16B"),
            Self::Bfp8B => write!(f, "Bfp8B"),
            Self::Bfp4B => write!(f, "Bfp4B"),
            Self::Bfp2B => write!(f, "Bfp2B"),
            Self::Lf8 => write!(f, "Lf8"),
            Self::Fp8E4M3 => write!(f, "Fp8E4M3"),
            Self::Int8 => write!(f, "Int8"),
            Self::Tf32 => write!(f, "Tf32"),
            Self::UInt8 => write!(f, "UInt8"),
            Self::UInt16 => write!(f, "UInt16"),
            Self::Int16 => write!(f, "Int16"),
            Self::Int32 => write!(f, "Int32"),
            Self::UInt32 => write!(f, "UInt32"),
            Self::RawUInt8 => write!(f, "RawUInt8"),
            Self::RawUInt16 => write!(f, "RawUInt16"),
            Self::RawUInt32 => write!(f, "RawUInt32"),
            Self::Invalid => write!(f, "Invalid"),
        }
    }
}

impl DataFormat {
    #[must_use]
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

    #[must_use]
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

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    #[must_use]
    pub fn ranges(&self) -> &[CoreRange] {
        &self.ranges
    }

    fn ffi_ranges(&self) -> Vec<ffi::CoreRangeRepr> {
        self.ranges.iter().copied().map(CoreRange::as_ffi).collect()
    }
}

impl IntoIterator for CoreRangeSet {
    type Item = CoreRange;
    type IntoIter = std::vec::IntoIter<CoreRange>;

    fn into_iter(self) -> Self::IntoIter {
        self.ranges.into_iter()
    }
}

impl FromIterator<CoreRange> for CoreRangeSet {
    fn from_iter<I: IntoIterator<Item = CoreRange>>(iter: I) -> Self {
        Self {
            ranges: iter.into_iter().collect(),
        }
    }
}

impl Extend<CoreRange> for CoreRangeSet {
    fn extend<I: IntoIterator<Item = CoreRange>>(&mut self, iter: I) {
        self.ranges.extend(iter);
    }
}

impl From<CoreRange> for CoreRangeSet {
    fn from(range: CoreRange) -> Self {
        Self::from_range(range)
    }
}

impl From<LogicalCore> for CoreRangeSet {
    fn from(core: LogicalCore) -> Self {
        Self::from_core(core)
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

    #[must_use]
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

    #[must_use]
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

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("address", &self.address())
            .field("size", &self.size())
            .field("page_size", &self.page_size())
            .field("buffer_type", &self.buffer_type())
            .field("is_allocated", &self.is_allocated())
            .finish()
    }
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

    #[must_use]
    pub fn is_allocated(&self) -> bool {
        self.info().is_allocated
    }

    #[must_use]
    pub fn address(&self) -> u32 {
        self.info().address
    }

    #[must_use]
    pub fn size(&self) -> u64 {
        self.info().size
    }

    #[must_use]
    pub fn page_size(&self) -> u64 {
        self.info().page_size
    }

    #[must_use]
    pub fn buffer_type(&self) -> BufferType {
        BufferType::from_ffi(self.info().buffer_type)
    }

    #[must_use]
    pub fn buffer_layout(&self) -> TensorMemoryLayout {
        TensorMemoryLayout::from_ffi(self.info().buffer_layout)
    }

    #[must_use]
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

impl std::fmt::Debug for CircularBufferConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircularBufferConfig")
            .finish_non_exhaustive()
    }
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

impl std::fmt::Debug for CircularBufferIndex<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircularBufferIndex")
            .field("buffer_index", &self.buffer_index)
            .field("remote", &self.remote)
            .finish()
    }
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
    use super::*;

    #[test]
    fn buffer_type_display() {
        assert_eq!(BufferType::Dram.to_string(), "DRAM");
        assert_eq!(BufferType::L1.to_string(), "L1");
        assert_eq!(BufferType::SystemMemory.to_string(), "SystemMemory");
        assert_eq!(BufferType::L1Small.to_string(), "L1Small");
        assert_eq!(BufferType::Trace.to_string(), "Trace");
    }

    #[test]
    fn buffer_type_as_ffi_full() {
        assert_eq!(BufferType::Dram.as_ffi(), 0);
        assert_eq!(BufferType::L1.as_ffi(), 1);
        assert_eq!(BufferType::SystemMemory.as_ffi(), 2);
        assert_eq!(BufferType::L1Small.as_ffi(), 3);
        assert_eq!(BufferType::Trace.as_ffi(), 4);
    }

    #[test]
    fn buffer_type_from_ffi_round_trip() {
        for (variant, ffi_val) in [
            (BufferType::Dram, 0u8),
            (BufferType::L1, 1),
            (BufferType::SystemMemory, 2),
            (BufferType::L1Small, 3),
            (BufferType::Trace, 4),
        ] {
            assert_eq!(BufferType::from_ffi(ffi_val), variant);
            assert_eq!(BufferType::from_ffi(variant.as_ffi()), variant);
        }
    }

    #[test]
    fn tensor_memory_layout_display() {
        assert_eq!(TensorMemoryLayout::Interleaved.to_string(), "Interleaved");
        assert_eq!(
            TensorMemoryLayout::HeightSharded.to_string(),
            "HeightSharded"
        );
        assert_eq!(TensorMemoryLayout::WidthSharded.to_string(), "WidthSharded");
        assert_eq!(TensorMemoryLayout::BlockSharded.to_string(), "BlockSharded");
        assert_eq!(TensorMemoryLayout::NdSharded.to_string(), "NdSharded");
    }

    #[test]
    fn tensor_memory_layout_as_ffi_full() {
        assert_eq!(TensorMemoryLayout::Interleaved.as_ffi(), 0);
        assert_eq!(TensorMemoryLayout::HeightSharded.as_ffi(), 2);
        assert_eq!(TensorMemoryLayout::WidthSharded.as_ffi(), 3);
        assert_eq!(TensorMemoryLayout::BlockSharded.as_ffi(), 4);
        assert_eq!(TensorMemoryLayout::NdSharded.as_ffi(), 5);
    }

    #[test]
    fn tensor_memory_layout_from_ffi_round_trip() {
        for (variant, ffi_val) in [
            (TensorMemoryLayout::Interleaved, 0u8),
            (TensorMemoryLayout::HeightSharded, 2),
            (TensorMemoryLayout::WidthSharded, 3),
            (TensorMemoryLayout::BlockSharded, 4),
            (TensorMemoryLayout::NdSharded, 5),
        ] {
            assert_eq!(TensorMemoryLayout::from_ffi(ffi_val), variant);
            assert_eq!(TensorMemoryLayout::from_ffi(variant.as_ffi()), variant);
        }
    }

    #[test]
    fn shard_orientation_display() {
        assert_eq!(ShardOrientation::RowMajor.to_string(), "RowMajor");
        assert_eq!(ShardOrientation::ColMajor.to_string(), "ColMajor");
    }

    #[test]
    fn shard_orientation_as_ffi() {
        assert_eq!(ShardOrientation::RowMajor.as_ffi(), 0);
        assert_eq!(ShardOrientation::ColMajor.as_ffi(), 1);
    }

    #[test]
    fn shard_orientation_from_ffi_round_trip() {
        assert_eq!(ShardOrientation::from_ffi(0), ShardOrientation::RowMajor);
        assert_eq!(ShardOrientation::from_ffi(1), ShardOrientation::ColMajor);
        assert_eq!(
            ShardOrientation::from_ffi(ShardOrientation::RowMajor.as_ffi()),
            ShardOrientation::RowMajor
        );
        assert_eq!(
            ShardOrientation::from_ffi(ShardOrientation::ColMajor.as_ffi()),
            ShardOrientation::ColMajor
        );
    }

    #[test]
    fn data_format_display_selection() {
        assert_eq!(DataFormat::Float32.to_string(), "Float32");
        assert_eq!(DataFormat::Bfp2.to_string(), "Bfp2");
        assert_eq!(DataFormat::Fp8E4M3.to_string(), "Fp8E4M3");
        assert_eq!(DataFormat::RawUInt8.to_string(), "RawUInt8");
        assert_eq!(DataFormat::Invalid.to_string(), "Invalid");
    }

    #[test]
    fn data_format_from_ffi_round_trip() {
        for variant in [
            DataFormat::Float32,
            DataFormat::Float16,
            DataFormat::Bfp8,
            DataFormat::Bfp4,
            DataFormat::Bfp2,
            DataFormat::Float16B,
            DataFormat::Bfp8B,
            DataFormat::Bfp4B,
            DataFormat::Bfp2B,
            DataFormat::Lf8,
            DataFormat::Fp8E4M3,
            DataFormat::Int8,
            DataFormat::Tf32,
            DataFormat::UInt8,
            DataFormat::UInt16,
            DataFormat::Int16,
            DataFormat::Int32,
            DataFormat::UInt32,
            DataFormat::RawUInt8,
            DataFormat::RawUInt16,
            DataFormat::RawUInt32,
            DataFormat::Invalid,
        ] {
            assert_eq!(DataFormat::from_ffi(variant.as_ffi()), variant);
        }
    }

    #[test]
    fn core_range_from_core_is_single_point() {
        let core = LogicalCore::new(2, 3);
        let range = CoreRange::from_core(core);
        assert_eq!(range.start, core);
        assert_eq!(range.end, core);
    }

    #[test]
    fn core_range_new_explicit() {
        let range = CoreRange::new(LogicalCore::new(0, 0), LogicalCore::new(1, 2));
        assert_eq!(range.start, LogicalCore::new(0, 0));
        assert_eq!(range.end, LogicalCore::new(1, 2));
    }

    #[test]
    fn core_range_set_default_is_empty() {
        let set = CoreRangeSet::default();
        assert!(set.is_empty());
        assert!(set.ranges().is_empty());
    }

    #[test]
    fn core_range_set_new_is_empty() {
        assert!(CoreRangeSet::new().is_empty());
    }

    #[test]
    fn core_range_set_from_core() {
        let set = CoreRangeSet::from_core(LogicalCore::new(0, 0));
        assert!(!set.is_empty());
        assert_eq!(set.ranges().len(), 1);
        assert_eq!(set.ranges()[0].start, LogicalCore::new(0, 0));
        assert_eq!(set.ranges()[0].end, LogicalCore::new(0, 0));
    }

    #[test]
    fn core_range_set_from_range() {
        let range = CoreRange::from_core(LogicalCore::new(1, 2));
        let set = CoreRangeSet::from_range(range);
        assert_eq!(set.ranges().len(), 1);
        assert_eq!(set.ranges()[0], range);
    }

    #[test]
    fn core_range_set_from_ranges() {
        let set = CoreRangeSet::from_ranges([
            CoreRange::from_core(LogicalCore::new(0, 0)),
            CoreRange::from_core(LogicalCore::new(1, 1)),
        ]);
        assert_eq!(set.ranges().len(), 2);
    }

    #[test]
    fn core_range_set_push_extends() {
        let mut set = CoreRangeSet::new();
        assert!(set.is_empty());
        set.push(CoreRange::from_core(LogicalCore::new(0, 0)));
        assert!(!set.is_empty());
        assert_eq!(set.ranges().len(), 1);
        set.push(CoreRange::from_core(LogicalCore::new(1, 1)));
        assert_eq!(set.ranges().len(), 2);
    }

    #[test]
    fn core_range_set_from_iterator() {
        let set: CoreRangeSet = [
            CoreRange::from_core(LogicalCore::new(0, 0)),
            CoreRange::from_core(LogicalCore::new(1, 1)),
        ]
        .into_iter()
        .collect();
        assert_eq!(set.ranges().len(), 2);
    }

    #[test]
    fn core_range_set_extend() {
        let mut set = CoreRangeSet::from_core(LogicalCore::new(0, 0));
        set.extend([CoreRange::from_core(LogicalCore::new(1, 1))]);
        assert_eq!(set.ranges().len(), 2);
    }

    #[test]
    fn core_range_set_from_core_range() {
        let range = CoreRange::from_core(LogicalCore::new(3, 4));
        let set = CoreRangeSet::from(range);
        assert_eq!(set.ranges().len(), 1);
        assert_eq!(set.ranges()[0], range);
    }

    #[test]
    fn core_range_set_from_logical_core() {
        let set = CoreRangeSet::from(LogicalCore::new(5, 6));
        assert_eq!(set.ranges().len(), 1);
        assert_eq!(set.ranges()[0].start, LogicalCore::new(5, 6));
    }

    #[test]
    fn core_range_set_into_iter() {
        let set = CoreRangeSet::from_ranges([
            CoreRange::from_core(LogicalCore::new(0, 0)),
            CoreRange::from_core(LogicalCore::new(1, 1)),
        ]);
        let mut count = 0;
        for _range in set {
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn shard_spec_raw_values() {
        let spec = ShardSpec::new([8, 16], ShardOrientation::ColMajor);
        assert_eq!(spec.shape, [8, 16]);
        assert_eq!(spec.orientation, ShardOrientation::ColMajor);
    }

    #[test]
    fn shard_spec_buffer_construction() {
        let grid = CoreRangeSet::from_core(LogicalCore::new(0, 0));
        let buf = ShardSpecBuffer::new(
            grid.clone(),
            [2, 4],
            ShardOrientation::RowMajor,
            [1, 1],
            [2, 4],
        );
        assert_eq!(buf.grid, grid);
        assert_eq!(buf.shard_spec.shape, [2, 4]);
        assert_eq!(buf.shard_spec.orientation, ShardOrientation::RowMajor);
        assert_eq!(buf.page_shape, [1, 1]);
        assert_eq!(buf.tensor2d_shape_in_pages, [2, 4]);
    }

    #[test]
    fn tile_config_construction() {
        let tile = TileConfig::new(32, 32, false);
        assert_eq!(tile.height, 32);
        assert_eq!(tile.width, 32);
        assert!(!tile.transpose_tile);

        let transposed = TileConfig::new(16, 32, true);
        assert!(transposed.transpose_tile);
    }

    #[test]
    fn interleaved_buffer_config_construction() {
        let config = InterleavedBufferConfig::new(4096, 2048, BufferType::Dram);
        assert_eq!(config.size, 4096);
        assert_eq!(config.page_size, 2048);
        assert_eq!(config.buffer_type, BufferType::Dram);
    }

    #[test]
    fn sharded_buffer_config_construction() {
        let grid = CoreRangeSet::from_core(LogicalCore::new(0, 0));
        let shard_spec = ShardSpecBuffer::new(
            grid.clone(),
            [4, 8],
            ShardOrientation::ColMajor,
            [2, 2],
            [2, 4],
        );
        let config = ShardedBufferConfig::new(
            8192,
            1024,
            BufferType::L1,
            TensorMemoryLayout::WidthSharded,
            shard_spec.clone(),
        );
        assert_eq!(config.size, 8192);
        assert_eq!(config.page_size, 1024);
        assert_eq!(config.buffer_type, BufferType::L1);
        assert_eq!(config.buffer_layout, TensorMemoryLayout::WidthSharded);
        assert_eq!(config.shard_spec, shard_spec);
    }

    #[test]
    fn buffer_create_options_default_is_none() {
        let opts = BufferCreateOptions::new();
        assert!(opts.address.is_none());
        assert!(opts.sub_device_id.is_none());
    }

    #[test]
    fn buffer_create_options_with_address_only_ok() {
        let opts = BufferCreateOptions::new().with_address(0xDEAD_BEEF);
        assert_eq!(opts.address, Some(0xDEAD_BEEF));
        assert!(opts.sub_device_id.is_none());
        assert!(opts.as_ffi().is_ok());
    }

    #[test]
    fn buffer_create_options_with_sub_device_only_ok() {
        let opts = BufferCreateOptions::new().with_sub_device_id(SubDeviceId(7));
        assert!(opts.address.is_none());
        assert_eq!(opts.sub_device_id, Some(SubDeviceId(7)));
        assert!(opts.as_ffi().is_ok());
    }

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
