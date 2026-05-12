#pragma once

#include <cstdint>
#include <vector>

#include "rust/cxx.h"

namespace tt_metal_cxx {

/// Tilize: convert row-major host data to TT tilized (nfaces) layout.
///
/// `data` is raw bytes. `m` and `n` are element counts (must be multiples of 32).
/// `elem_size` is bytes per element (2 or 4).
///
/// Returns tilized data as bytes. The output may be larger than input if
/// `m` or `n` are not tile-aligned (padding to tile boundaries).
rust::Vec<uint8_t> tilize(
    rust::Slice<const uint8_t> data, uint32_t m, uint32_t n, uint32_t elem_size);

/// Untilize: convert TT tilized (nfaces) layout back to row-major.
///
/// `data` is tilized bytes. `m` and `n` are element counts (must be multiples of 32).
/// `elem_size` is bytes per element (2 or 4).
///
/// Returns row-major data as bytes, stripped of tile padding.
rust::Vec<uint8_t> untilize(
    rust::Slice<const uint8_t> data, uint32_t m, uint32_t n, uint32_t elem_size);

}  // namespace tt_metal_cxx
