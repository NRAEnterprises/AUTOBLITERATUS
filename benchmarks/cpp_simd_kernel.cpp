#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace {
constexpr char kInputMagic[8] = {'A', 'B', 'S', 'M', 'D', '0', '1', '\0'};
constexpr char kOutputMagic[8] = {'A', 'B', 'S', 'O', 'U', 'T', '1', '\0'};
constexpr std::uint32_t kMaxElements = 1'000'000;

struct InputHeader {
    char magic[8];
    std::uint32_t count;
    float scale;
    float lower;
    float upper;
};

struct OutputHeader {
    char magic[8];
    std::uint32_t count;
};

bool finite(float value) { return std::isfinite(value); }

bool scalar_kernel(const float* input, float* output, std::size_t count, float scale, float lower, float upper) {
    for (std::size_t index = 0; index < count; ++index) {
        const float product = input[index] * scale;
        if (!finite(product)) {
            return false;
        }
        output[index] = std::min(upper, std::max(lower, product));
    }
    return true;
}

bool simd_kernel(const float* input, float* output, std::size_t count, float scale, float lower, float upper) {
    std::size_t index = 0;
#if defined(CPP_SIMD_FORCE_SCALAR)
    return scalar_kernel(input, output, count, scale, lower, upper);
#elif defined(__aarch64__)
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    const float32x4_t lower_vec = vdupq_n_f32(lower);
    const float32x4_t upper_vec = vdupq_n_f32(upper);
    for (; index + 4 <= count; index += 4) {
        float32x4_t values = vmulq_f32(vld1q_f32(input + index), scale_vec);
        values = vmaxq_f32(lower_vec, values);
        values = vminq_f32(upper_vec, values);
        vst1q_f32(output + index, values);
    }
#elif defined(__SSE2__)
    const __m128 scale_vec = _mm_set1_ps(scale);
    const __m128 lower_vec = _mm_set1_ps(lower);
    const __m128 upper_vec = _mm_set1_ps(upper);
    for (; index + 4 <= count; index += 4) {
        __m128 values = _mm_mul_ps(_mm_loadu_ps(input + index), scale_vec);
        values = _mm_max_ps(lower_vec, values);
        values = _mm_min_ps(upper_vec, values);
        _mm_storeu_ps(output + index, values);
    }
#endif
    return scalar_kernel(input + index, output + index, count - index, scale, lower, upper);
}

int fail(const std::string& message) {
    std::cerr << "cpp_simd_error:" << message << '\n';
    return 2;
}

int parse_positive(const char* value, const char* name, std::size_t maximum, std::size_t* result) {
    try {
        const auto parsed = std::stoull(value);
        if (parsed == 0 || parsed > maximum) {
            return fail(std::string("invalid_") + name);
        }
        *result = static_cast<std::size_t>(parsed);
        return 0;
    } catch (const std::exception&) {
        return fail(std::string("invalid_") + name);
    }
}

int benchmark(const char* mode, const char* length_text, const char* iterations_text) {
    std::size_t length = 0;
    std::size_t iterations = 0;
    if (parse_positive(length_text, "benchmark_length", 10'000'000, &length) != 0 ||
        parse_positive(iterations_text, "benchmark_iterations", 10'000'000, &iterations) != 0) {
        return 2;
    }
    if (std::strcmp(mode, "scalar") != 0 && std::strcmp(mode, "simd") != 0) {
        return fail("invalid_benchmark_mode");
    }
    std::vector<float> input(length);
    std::vector<float> output(length);
    for (std::size_t index = 0; index < length; ++index) {
        input[index] = static_cast<float>((static_cast<int>(index % 97) - 48) / 17.0);
    }
    const auto started = std::chrono::steady_clock::now();
    volatile float checksum = 0.0F;
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        const bool success = std::strcmp(mode, "scalar") == 0
                                 ? scalar_kernel(input.data(), output.data(), length, 1.25F, -3.0F, 3.0F)
                                 : simd_kernel(input.data(), output.data(), length, 1.25F, -3.0F, 3.0F);
        if (!success) {
            return fail("benchmark_kernel_failure");
        }
        checksum += output[iteration % length];
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started
    ).count();
    std::cout << "mode=" << mode << " length=" << length << " iterations=" << iterations
              << " elapsed_ns=" << elapsed << " checksum=" << checksum << '\n';
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 5 && std::strcmp(argv[1], "--benchmark") == 0) {
        return benchmark(argv[2], argv[3], argv[4]);
    }
    if (argc != 3) {
        return fail("usage: cpp_simd_kernel INPUT OUTPUT");
    }
    std::ifstream input_file(argv[1], std::ios::binary);
    if (!input_file) {
        return fail("input_open");
    }
    InputHeader header{};
    input_file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input_file || std::memcmp(header.magic, kInputMagic, sizeof(kInputMagic)) != 0) {
        return fail("invalid_input_header");
    }
    if (header.count == 0 || header.count > kMaxElements) {
        return fail("invalid_element_count");
    }
    if (!finite(header.scale) || !finite(header.lower) || !finite(header.upper) || header.lower > header.upper) {
        return fail("invalid_parameters");
    }
    std::vector<float> values(header.count);
    input_file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!input_file) {
        return fail("truncated_input");
    }
    char extra{};
    if (input_file.read(&extra, 1)) {
        return fail("trailing_input");
    }
    for (float value : values) {
        if (!finite(value)) {
            return fail("nonfinite_input");
        }
    }
    std::vector<float> output(header.count);
    if (!simd_kernel(values.data(), output.data(), values.size(), header.scale, header.lower, header.upper)) {
        return fail("float32_overflow");
    }
    for (float value : output) {
        if (!finite(value)) {
            return fail("nonfinite_output");
        }
    }
    std::ofstream output_file(argv[2], std::ios::binary | std::ios::trunc);
    if (!output_file) {
        return fail("output_open");
    }
    OutputHeader output_header{};
    std::memcpy(output_header.magic, kOutputMagic, sizeof(kOutputMagic));
    output_header.count = header.count;
    output_file.write(reinterpret_cast<const char*>(&output_header), sizeof(output_header));
    output_file.write(reinterpret_cast<const char*>(output.data()), static_cast<std::streamsize>(output.size() * sizeof(float)));
    if (!output_file) {
        return fail("output_write");
    }
    return 0;
}
