#pragma once

#include <hnswlib/hnswlib.h>

#include <cmath>

namespace hnswlib {


static float
    CorrelationDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    float* v1 = (float*)pVect1;
    float* v2 = (float*)pVect2;

    // Calculate means
    float mu_x = 0.0f;
    float mu_y = 0.0f;
    for (size_t i = 0; i < qty; i++) {
        mu_x += v1[i];
        mu_y += v2[i];
    }
    mu_x /= qty;
    mu_y /= qty;

    // Calculate correlation components
    float norm_x = 0.0f;
    float norm_y = 0.0f;
    float dot_product = 0.0f;

    for (size_t i = 0; i < qty; i++) {
        float shifted_x = v1[i] - mu_x;
        float shifted_y = v2[i] - mu_y;
        norm_x += shifted_x * shifted_x;
        norm_y += shifted_y * shifted_y;
        dot_product += shifted_x * shifted_y;
    }

    if (norm_x == 0.0f && norm_y == 0.0f) {
        return 0.0f;
    }
    else if (dot_product == 0.0f) {
        return 1.0f;
    }
    else {
        return 1.0f - (dot_product / std::sqrt(norm_x * norm_y));
    }
}

#if defined(USE_AVX)

static float
CorrelationDistanceSIMD8ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty8 = qty / 8;
    const float* pEnd1 = pVect1 + 8 * qty8;

    // Calculate means using SIMD
    __m256 sum1 = _mm256_set1_ps(0.0f);
    __m256 sum2 = _mm256_set1_ps(0.0f);

    float* p1 = pVect1;
    float* p2 = pVect2;

    while (p1 < pEnd1) {
        __m256 v1 = _mm256_loadu_ps(p1);
        __m256 v2 = _mm256_loadu_ps(p2);
        sum1 = _mm256_add_ps(sum1, v1);
        sum2 = _mm256_add_ps(sum2, v2);
        p1 += 8;
        p2 += 8;
    }

    // Extract and sum the elements
    float PORTABLE_ALIGN32 tmp1[8], tmp2[8];
    _mm256_store_ps(tmp1, sum1);
    _mm256_store_ps(tmp2, sum2);

    float mu_x = 0.0f, mu_y = 0.0f;
    for (int i = 0; i < 8; i++) {
        mu_x += tmp1[i];
        mu_y += tmp2[i];
    }

    // Add remaining elements
    for (size_t i = qty8 * 8; i < qty; i++) {
        mu_x += pVect1[i];
        mu_y += pVect2[i];
    }

    mu_x /= qty;
    mu_y /= qty;

    // Calculate correlation components using SIMD
    __m256 mu_x_vec = _mm256_set1_ps(mu_x);
    __m256 mu_y_vec = _mm256_set1_ps(mu_y);
    __m256 norm_x_vec = _mm256_set1_ps(0.0f);
    __m256 norm_y_vec = _mm256_set1_ps(0.0f);
    __m256 dot_vec = _mm256_set1_ps(0.0f);

    p1 = pVect1;
    p2 = pVect2;

    while (p1 < pEnd1) {
        __m256 v1 = _mm256_loadu_ps(p1);
        __m256 v2 = _mm256_loadu_ps(p2);

        __m256 shifted_x = _mm256_sub_ps(v1, mu_x_vec);
        __m256 shifted_y = _mm256_sub_ps(v2, mu_y_vec);

        norm_x_vec = _mm256_add_ps(norm_x_vec, _mm256_mul_ps(shifted_x, shifted_x));
        norm_y_vec = _mm256_add_ps(norm_y_vec, _mm256_mul_ps(shifted_y, shifted_y));
        dot_vec = _mm256_add_ps(dot_vec, _mm256_mul_ps(shifted_x, shifted_y));

        p1 += 8;
        p2 += 8;
    }

    // Extract results
    float PORTABLE_ALIGN32 norm_x_tmp[8], norm_y_tmp[8], dot_tmp[8];
    _mm256_store_ps(norm_x_tmp, norm_x_vec);
    _mm256_store_ps(norm_y_tmp, norm_y_vec);
    _mm256_store_ps(dot_tmp, dot_vec);

    float norm_x = 0.0f, norm_y = 0.0f, dot_product = 0.0f;
    for (int i = 0; i < 8; i++) {
        norm_x += norm_x_tmp[i];
        norm_y += norm_y_tmp[i];
        dot_product += dot_tmp[i];
    }

    // Handle remaining elements
    for (size_t i = qty8 * 8; i < qty; i++) {
        float shifted_x = pVect1[i] - mu_x;
        float shifted_y = pVect2[i] - mu_y;
        norm_x += shifted_x * shifted_x;
        norm_y += shifted_y * shifted_y;
        dot_product += shifted_x * shifted_y;
    }

    if (norm_x == 0.0f && norm_y == 0.0f) {
        return 0.0f;
    }
    else if (dot_product == 0.0f) {
        return 1.0f;
    }
    else {
        return 1.0f - (dot_product / std::sqrt(norm_x * norm_y));
    }
}

#endif // USE_AVX

#if defined(USE_SSE)

static float
CorrelationDistanceSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty4 = qty / 4;
    const float* pEnd1 = pVect1 + 4 * qty4;

    // Calculate means using SIMD
    __m128 sum1 = _mm_set1_ps(0.0f);
    __m128 sum2 = _mm_set1_ps(0.0f);

    float* p1 = pVect1;
    float* p2 = pVect2;

    while (p1 < pEnd1) {
        __m128 v1 = _mm_loadu_ps(p1);
        __m128 v2 = _mm_loadu_ps(p2);
        sum1 = _mm_add_ps(sum1, v1);
        sum2 = _mm_add_ps(sum2, v2);
        p1 += 4;
        p2 += 4;
    }

    // Extract and sum the elements
    float PORTABLE_ALIGN32 tmp1[4], tmp2[4];
    _mm_store_ps(tmp1, sum1);
    _mm_store_ps(tmp2, sum2);

    float mu_x = tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3];
    float mu_y = tmp2[0] + tmp2[1] + tmp2[2] + tmp2[3];

    // Add remaining elements
    for (size_t i = qty4 * 4; i < qty; i++) {
        mu_x += pVect1[i];
        mu_y += pVect2[i];
    }

    mu_x /= qty;
    mu_y /= qty;

    // Calculate correlation components using SIMD
    __m128 mu_x_vec = _mm_set1_ps(mu_x);
    __m128 mu_y_vec = _mm_set1_ps(mu_y);
    __m128 norm_x_vec = _mm_set1_ps(0.0f);
    __m128 norm_y_vec = _mm_set1_ps(0.0f);
    __m128 dot_vec = _mm_set1_ps(0.0f);

    p1 = pVect1;
    p2 = pVect2;

    while (p1 < pEnd1) {
        __m128 v1 = _mm_loadu_ps(p1);
        __m128 v2 = _mm_loadu_ps(p2);

        __m128 shifted_x = _mm_sub_ps(v1, mu_x_vec);
        __m128 shifted_y = _mm_sub_ps(v2, mu_y_vec);

        norm_x_vec = _mm_add_ps(norm_x_vec, _mm_mul_ps(shifted_x, shifted_x));
        norm_y_vec = _mm_add_ps(norm_y_vec, _mm_mul_ps(shifted_y, shifted_y));
        dot_vec = _mm_add_ps(dot_vec, _mm_mul_ps(shifted_x, shifted_y));

        p1 += 4;
        p2 += 4;
    }

    // Extract results
    float PORTABLE_ALIGN32 norm_x_tmp[4], norm_y_tmp[4], dot_tmp[4];
    _mm_store_ps(norm_x_tmp, norm_x_vec);
    _mm_store_ps(norm_y_tmp, norm_y_vec);
    _mm_store_ps(dot_tmp, dot_vec);

    float norm_x = norm_x_tmp[0] + norm_x_tmp[1] + norm_x_tmp[2] + norm_x_tmp[3];
    float norm_y = norm_y_tmp[0] + norm_y_tmp[1] + norm_y_tmp[2] + norm_y_tmp[3];
    float dot_product = dot_tmp[0] + dot_tmp[1] + dot_tmp[2] + dot_tmp[3];

    // Handle remaining elements
    for (size_t i = qty4 * 4; i < qty; i++) {
        float shifted_x = pVect1[i] - mu_x;
        float shifted_y = pVect2[i] - mu_y;
        norm_x += shifted_x * shifted_x;
        norm_y += shifted_y * shifted_y;
        dot_product += shifted_x * shifted_y;
    }

    if (norm_x == 0.0f && norm_y == 0.0f) {
        return 0.0f;
    }
    else if (dot_product == 0.0f) {
        return 1.0f;
    }
    else {
        return 1.0f - (dot_product / std::sqrt(norm_x * norm_y));
    }
}

#endif // USE_SSE


class CorrelationSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    CorrelationSpace(size_t dim) {

        fstdistfunc_ = CorrelationDistance;

#if defined(USE_SSE)
        if (dim > 4)
            fstdistfunc_ = CorrelationDistanceSIMD4ExtSSE;
#endif

#if defined(USE_AVX)
        if (dim > 8)
            fstdistfunc_ = CorrelationDistanceSIMD8ExtAVX;
#endif

        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

~CorrelationSpace() {}
};

}  // namespace hnswlib
