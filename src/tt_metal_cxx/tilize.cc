#include "tt_metal_cxx/tilize.hpp"

#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>

namespace tt_metal_cxx {

namespace {

// Helper: reinterpret Vec<T> as Vec<uint8_t>
template <typename T>
rust::Vec<uint8_t> vec_to_bytes(const std::vector<T>& vec) {
    rust::Vec<uint8_t> out;
    auto* bytes = reinterpret_cast<const uint8_t*>(vec.data());
    auto byte_size = vec.size() * sizeof(T);
    out.reserve(byte_size);
    for (size_t i = 0; i < byte_size; i++) {
        out.push_back(bytes[i]);
    }
    return out;
}

// Helper: reinterpret uint8_t slice as Vec<T> (expects correct alignment and size)
template <typename T>
std::vector<T> bytes_to_vec(rust::Slice<const uint8_t> data, size_t num_elements) {
    auto* ptr = reinterpret_cast<const T*>(data.data());
    return std::vector<T>(ptr, ptr + num_elements);
}

template <typename T>
rust::Vec<uint8_t> tilize_impl(rust::Slice<const uint8_t> data, uint32_t m, uint32_t n) {
    auto input = bytes_to_vec<T>(data, m * n);
    auto result = ::tilize_nfaces(input, m, n);
    return vec_to_bytes(result);
}

template <typename T>
rust::Vec<uint8_t> untilize_impl(rust::Slice<const uint8_t> data, uint32_t m, uint32_t n) {
    auto input = bytes_to_vec<T>(data, m * n);
    auto result = ::untilize_nfaces(input, m, n);
    return vec_to_bytes(result);
}

}  // namespace

rust::Vec<uint8_t> tilize(
    rust::Slice<const uint8_t> data, uint32_t m, uint32_t n, uint32_t elem_size) {
    if (m == 0 || n == 0 || data.empty()) {
        throw std::invalid_argument("m, n, and data must be non-empty");
    }
    if (m % 32 != 0 || n % 32 != 0) {
        throw std::invalid_argument("m and n must be multiples of 32 for tile alignment");
    }
    size_t num_elements = static_cast<size_t>(m) * n;
    if (data.size() != num_elements * elem_size) {
        throw std::invalid_argument("data size does not match m * n * elem_size");
    }

    switch (elem_size) {
        case 2: return tilize_impl<bfloat16>(data, m, n);
        case 4: return tilize_impl<float>(data, m, n);
        default:
            throw std::invalid_argument("elem_size must be 2 or 4");
    }
}

rust::Vec<uint8_t> untilize(
    rust::Slice<const uint8_t> data, uint32_t m, uint32_t n, uint32_t elem_size) {
    if (m == 0 || n == 0 || data.empty()) {
        throw std::invalid_argument("m, n, and data must be non-empty");
    }
    if (m % 32 != 0 || n % 32 != 0) {
        throw std::invalid_argument("m and n must be multiples of 32 for tile alignment");
    }
    size_t num_elements = static_cast<size_t>(m) * n;
    if (data.size() != num_elements * elem_size) {
        throw std::invalid_argument("data size does not match m * n * elem_size");
    }

    switch (elem_size) {
        case 2: return untilize_impl<bfloat16>(data, m, n);
        case 4: return untilize_impl<float>(data, m, n);
        default:
            throw std::invalid_argument("elem_size must be 2 or 4");
    }
}

}  // namespace tt_metal_cxx
