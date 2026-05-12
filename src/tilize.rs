use crate::Exception;
use crate::ffi::ffi;

/// Convert row-major host data to TT-Metal tilized (nfaces) layout.
///
/// `m` and `n` are the logical element dimensions of the tensor
/// (must be multiples of 32).
/// `elem_size` is the byte size of each element (1, 2, or 4).
///
/// Returns tilized data, padded to tile boundaries.
pub fn tilize(data: &[u8], m: u32, n: u32, elem_size: u32) -> Result<Vec<u8>, Exception> {
    ffi::tilize(data, m, n, elem_size)
}

/// Convert TT-Metal tilized (nfaces) data back to row-major layout.
///
/// `m` and `n` are the logical element dimensions of the tensor
/// (must be multiples of 32).
/// `elem_size` is the byte size of each element (1, 2, or 4).
///
/// Returns row-major data, stripped of tile padding.
pub fn untilize(data: &[u8], m: u32, n: u32, elem_size: u32) -> Result<Vec<u8>, Exception> {
    ffi::untilize(data, m, n, elem_size)
}
