//
//  PackedFunction.cpp
//  MNN
//
//  Created by MNN on b'2021/07/05'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <float.h>
#include <string.h>
#include <algorithm>
#include <limits>
#include <vector>
#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "backend/cpu/CPUPool.hpp"
#include "backend/cpu/BinaryUtils.hpp"
#include "Vec8.hpp"
#define PACK_UNIT 8
extern "C" {
void _AVX_MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);
void _AVX_MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);
void _AVX_MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber, size_t biasNumber);
void _AVX_MNNDeconvRunForUnitDepthWise(const float* dst, float* src, const float* weight, size_t fw, size_t fh,
                                       size_t weight_y_step, size_t dilateX_step, size_t dilateY_step);
void _AVX_MNNDeconvRunForLineDepthwise(const float* dst, float* src, const float* weight, size_t width, size_t src_w_setup,
                                       size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step);
void _AVX_MNNGridSampleInterp(float* outputPtr, const float* inputPtr, const float* cordPtr, size_t inH, size_t inW, size_t outW, bool sampleMode, bool padMode);
void _AVX_MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride, size_t eSub, size_t hSub);
void _AVX_MNNConvRunForUnitDepthWise(float* dst, const float* src, const float* weight, size_t fw, size_t fh,
                                     size_t weight_y_step, size_t dilateX_step, size_t dilateY_step);
void _AVX_MNNMultiAndDestTransformCommon23(float **cacheLine, const float *weigth, float *dest, int cacheLineSize, int ow, const float* bias, const float* parameter);
void _AVX_MNNSourceTransformCommonF23(const float *source, float *dest, int unit, int iw, int pad, int su, int eu);
void _AVX_MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow, const float* bias, const float* parameter);
void _AVX_MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height);
void _AVX_MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height);
void _AVX_MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride,
                                    size_t length, size_t hSub);
void _AVX_MNNConvRunForLineDepthwise(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                     size_t srcHStep, size_t dstHStep);
void _AVX_MNNAxByClampBroadcastUnit(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters);
}


void _AVX_MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm256_storeu_ps(d, _mm256_loadu_ps(s));
    }
}
void _AVX_MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm256_storeu_ps(d, _mm256_add_ps(_mm256_loadu_ps(s), _mm256_loadu_ps(d)));
    }
}

void _AVX_MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    auto zero = _mm_set1_ps(0.0f);
    auto zero2 = _mm256_set1_ps(0.0f);
    int sizeC8 = sizeQuad;
    for (int j = 0; j < depthQuad; j++) {
        auto slopeZ       = _mm256_loadu_ps(slope + PACK_UNIT * j);
        const float* srcZ = src + PACK_UNIT * j * sizeQuad;
        float* dstZ       = dst + PACK_UNIT * j * sizeQuad;
        for (int i = 0; i < sizeC8; i++) {
            auto src   = _mm256_loadu_ps(srcZ);
            auto mask0 = _mm256_cmp_ps(src, zero2, 0x01);
            auto mask1 = _mm256_cmp_ps(src, zero2, 0x0D);
            auto other = _mm256_mul_ps(src, slopeZ);
            _mm256_storeu_ps(dstZ, _mm256_add_ps(_mm256_and_ps(other, mask0), _mm256_and_ps(src, mask1)));
            srcZ += PACK_UNIT;
            dstZ += PACK_UNIT;
        }
    }
}

void _AVX_MNNAxByClampBroadcastUnit(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters) {
    auto minF = _mm256_broadcast_ss(parameters + 2);
    auto maxF = _mm256_broadcast_ss(parameters + 3);
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + PACK_UNIT * y;
        auto bv = _mm256_loadu_ps(b);
        auto c = C + cStride * y;
        for (int x = 0; x < width; ++x) {
            auto av = _mm256_loadu_ps(a);
            auto cv = _mm256_add_ps(av, bv);
            cv = _mm256_min_ps(cv, maxF);
            cv = _mm256_max_ps(cv, minF);
            _mm256_storeu_ps(c, cv);
            a += PACK_UNIT;
            c += PACK_UNIT;
        }
    }
}

void _AVX_MNNConvRunForUnitDepthWise(float* dst, const float* src, const float* weight, size_t fw, size_t fh,
                                  size_t weight_y_step, size_t dilateX_step, size_t dilateY_step) {
    int fx, fy;
    __m256 dstValue = _mm256_setzero_ps();
    const float* src_z    = src;
    const float* weight_z = weight;
    for (fy = 0; fy < fh; ++fy) {
        const float* src_y    = src_z + fy * dilateY_step;
        const float* weight_y = weight_z + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            const float* weight_x = weight_y + PACK_UNIT * fx;
            const float* src_x    = src_y + fx * dilateX_step;
            dstValue = _mm256_add_ps(dstValue, _mm256_mul_ps(_mm256_loadu_ps(src_x), _mm256_loadu_ps(weight_x)));
        }
    }
    _mm256_storeu_ps(dst, dstValue);
}

void _AVX_MNNConvRunForLineDepthwise(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                     size_t srcHStep, size_t dstHStep) {
    int dx, fx, fy;
    const int unit = 4;
    int widthUnit = width / unit;
    int widthRemain = width - widthUnit * unit;
    const float* weight_z = weight;
    for (int y = 0; y < height; ++y) {
        auto srcY = src + y * srcHStep;
        auto dstY = dst + y * dstHStep;
        for (dx = 0; dx < widthUnit; ++dx) {
            auto dstValue0 = _mm256_setzero_ps();
            auto dstValue1 = _mm256_setzero_ps();
            auto dstValue2 = _mm256_setzero_ps();
            auto dstValue3 = _mm256_setzero_ps();
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = srcY + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * PACK_UNIT;
                for (fx = 0; fx < fw; ++fx) {
                    const float* src_x    = src_y + fx * dilateX_step;
                    const float* weight_x = weight_y + PACK_UNIT * fx;
                    auto weightValue = _mm256_loadu_ps(weight_x);
                    dstValue0 = _mm256_add_ps(dstValue0, _mm256_mul_ps(_mm256_loadu_ps(src_x + 0 * src_w_setup), weightValue));
                    dstValue1 = _mm256_add_ps(dstValue1, _mm256_mul_ps(_mm256_loadu_ps(src_x + 1 * src_w_setup), weightValue));
                    dstValue2 = _mm256_add_ps(dstValue2, _mm256_mul_ps(_mm256_loadu_ps(src_x + 2 * src_w_setup), weightValue));
                    dstValue3 = _mm256_add_ps(dstValue3, _mm256_mul_ps(_mm256_loadu_ps(src_x + 3 * src_w_setup), weightValue));
                }
            }
            _mm256_storeu_ps(dstY + PACK_UNIT * 0, dstValue0);
            _mm256_storeu_ps(dstY + PACK_UNIT * 1, dstValue1);
            _mm256_storeu_ps(dstY + PACK_UNIT * 2, dstValue2);
            _mm256_storeu_ps(dstY + PACK_UNIT * 3, dstValue3);
            dstY += PACK_UNIT * unit;
            srcY += unit * src_w_setup;
        }
        for (dx = 0; dx < widthRemain; ++dx) {
            float* dst_x          = dstY + dx * PACK_UNIT;
            auto dstValue = _mm256_setzero_ps();
            const float* src_z    = srcY + src_w_setup * dx;
            const float* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = src_z + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * PACK_UNIT;
                for (fx = 0; fx < fw; ++fx) {
                    const float* weight_x = weight_y + PACK_UNIT * fx;
                    const float* src_x    = src_y + fx * dilateX_step;
                    dstValue = _mm256_add_ps(dstValue, _mm256_mul_ps(_mm256_loadu_ps(src_x), _mm256_loadu_ps(weight_x)));
                }
            }
            _mm256_storeu_ps(dst_x, dstValue);
        }
    }
}

static MNNBinaryExecute _AVX2_MNNSelectBinaryFunctionForFloat(int opType) {
    auto vecF = MNN::selectVector<Vec8, 8>(opType);
    if (nullptr != vecF) {
        return vecF;
    }
    return MNN::MNNGetCoreFunctions()->MNNSelectBinaryFunctionForFloat(opType);
}

static void _8BitcopyWithStrideC4(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (float*)srcO;
    auto dst = (float*)dstO;
    for (int i=0; i<size; ++i) {
        _mm256_storeu_ps(dst, _mm256_loadu_ps(src));
        src+= (8 * stride);
        dst+= (8 * ds);
    }
}
static MNNCopyWithStride _selectBlit(int bytesC4) {
    if (32 == bytesC4) {
        return _8BitcopyWithStrideC4;
    }
    return nullptr;
}



void _AVX_MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                        size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        float* dstZ         = dst + planeNumber * PACK_UNIT * z;
        const float* srcZ   = src + planeNumber * PACK_UNIT * z;
        auto biasZ = Vec8::load(bias + PACK_UNIT * z);
        auto alphaZ = Vec8::load(alpha + PACK_UNIT * z);
        for (int p = 0; p < planeNumber; ++p) {
            float* dstX       = dstZ + PACK_UNIT * p;
            const float* srcX = srcZ + PACK_UNIT * p;
            Vec8::save(dstX, (Vec8::load(srcX) * alphaZ) + biasZ);
        }
    }
}

void _AVX_MNNDeconvRunForUnitDepthWise(const float* dst, float* src, const float* weight, size_t fw, size_t fh,
                                  size_t weight_y_step, size_t dilateX_step, size_t dilateY_step) {
    int fx, fy;
    float* src_z          = src;
    const float* weight_z = weight;
    Vec8 dstV             = Vec8::load(dst);
    for (fy = 0; fy < fh; ++fy) {
        float* src_y          = src_z + fy * dilateY_step;
        const float* weight_y = weight_z + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            Vec8 weight_x = Vec8::load(weight_y + PACK_UNIT * fx);
            Vec8 src_x    = Vec8::load(src_y + fx * dilateX_step);
            Vec8::save(src_y + fx * dilateX_step, src_x + weight_x * dstV);
        }
    }
}
void _AVX_MNNDeconvRunForLineDepthwise(const float* dst, float* src, const float* weight, size_t width, size_t src_w_setup,
                                  size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step) {
    int dx;
    for (dx = 0; dx < width; ++dx) {
        const float* dst_x = dst + dx * PACK_UNIT;
        float* src_dx      = src + src_w_setup * dx;
        _AVX_MNNDeconvRunForUnitDepthWise(dst_x, src_dx, weight, fw, fh, fw * PACK_UNIT, dilateX_step, dilateY_step);
    }
}

static __m256 MNNGridSampleLoadSample(int h, int w, const float *buffer, int height, int width, bool padMode) {
    if (h < 0 || h >= height || w < 0 || w >= width) {
        if(padMode == true) { //padMode == BorderMode_ZEROS
            return _mm256_setzero_ps();
        }
        // Clearly, CLAMP is the right way to go for GridSamplePaddingMode_BORDER
        // For GridSamplePaddingMode_REFLECTION, since we have reflected the values into (-1, 1),
        // the leftover reflections degrade to GridSamplePaddingMode_BORDER
        h = h < 0 ? 0 : ( h > (height - 1) ? (height - 1) : h);
        w = w < 0 ? 0 : ( w > (width - 1) ? (width - 1) : w);
    }

    return _mm256_loadu_ps(buffer + h * width * PACK_UNIT + w * PACK_UNIT);
}
void _AVX_MNNGridSampleInterp(float* outputPtr, const float* inputPtr, const float* cordPtr, size_t inH, size_t inW, size_t outW, bool sampleMode, bool padMode) {
    for (auto ow = 0; ow < outW; ++ow) {
        auto w = cordPtr[2 * ow + 0];
        auto h = cordPtr[2 * ow + 1];
        __m256 interp;

        if (sampleMode == true) { //sampleMode == SampleMode_NEAREST
            int nh = ::floor(h + 0.5f);
            int nw = ::floor(w + 0.5f);
            interp = MNNGridSampleLoadSample(nh, nw, inputPtr, inH, inW, padMode);
        } else { //sampleMode == GridSampleMode_BILINEAR
            int w0_h = ::floor(h);
            int w0_w = ::floor(w);
            int w1_h = ::ceil(h);
            int w1_w = ::ceil(w);
            auto oneV = _mm256_set1_ps(1.0f);

            __m256 i00 = MNNGridSampleLoadSample(w0_h, w0_w, inputPtr, inH, inW, padMode);
            __m256 i01 = MNNGridSampleLoadSample(w0_h, w1_w, inputPtr, inH, inW, padMode);
            __m256 i10 = MNNGridSampleLoadSample(w1_h, w0_w, inputPtr, inH, inW, padMode);
            __m256 i11 = MNNGridSampleLoadSample(w1_h, w1_w, inputPtr, inH, inW, padMode);
            auto f0 = _mm256_set1_ps((float)w1_w - w);
            auto f1 = _mm256_sub_ps(oneV, f0);
            auto h0 = _mm256_set1_ps((float)w1_h - h);
            auto h1 = _mm256_sub_ps(oneV, h0);

            __m256 i0 = _mm256_add_ps(_mm256_mul_ps(i00, f0), _mm256_mul_ps(i01, f1));
            __m256 i1 = _mm256_add_ps(_mm256_mul_ps(i10, f0), _mm256_mul_ps(i11, f1));
            interp = _mm256_add_ps(_mm256_mul_ps(i0, h0), _mm256_mul_ps(i1, h1));
        }

        _mm256_storeu_ps(outputPtr + PACK_UNIT * ow, interp);
    }
}

void _AVX_MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            _mm256_storeu_ps(c + PACK_UNIT * x, _mm256_add_ps(_mm256_loadu_ps(b + PACK_UNIT * x), _mm256_loadu_ps(a + PACK_UNIT * x)));
        }
    }
}

void _AVX_MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride, size_t eSub, size_t hSub) {
    const int unit = PACK_UNIT;
    for (int y=0; y<hSub; ++y) {
        auto c11Y = c11 + y * cStride;
        auto c12Y = c12 + y * cStride;
        auto c22Y = c22 + y * cStride;
        auto c21Y = c21 + y * cStride;
        auto xY = xAddr + y * eSub * unit;
        for (int x=0; x<eSub; ++x) {
            auto xv = _mm256_loadu_ps(xY + unit*x);
            auto c21v = _mm256_loadu_ps(c21Y + unit*x);
            auto c11v = _mm256_loadu_ps(c11Y + unit*x);
            auto c22v = _mm256_loadu_ps(c22Y + unit*x);
            auto c12v = _mm256_loadu_ps(c12Y + unit*x);
            c12v = _mm256_add_ps(c12v, xv);
            c21v = _mm256_add_ps(c12v, c21v);
            c12v = _mm256_add_ps(c22v, c12v);
            c22v = _mm256_add_ps(c22v, c21v);
            c12v = _mm256_add_ps(c11v, c12v);
            _mm256_storeu_ps(c12Y + unit*x, c12v);
            _mm256_storeu_ps(c22Y + unit*x, c22v);
            _mm256_storeu_ps(c21Y + unit*x, c21v);
        }
    }
}

void _AVX_MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            _mm256_storeu_ps(c + PACK_UNIT * x, _mm256_sub_ps(_mm256_loadu_ps(a + PACK_UNIT * x), _mm256_loadu_ps(b + PACK_UNIT * x)));
        }
    }
}

void _AVX_MNNMultiAndDestTransformCommon23(float **cacheLine, const float *weigth, float *dest, int cacheLineSize, int ow, const float* bias, const float* parameter) {
    int unit = ow / 2;
    MNN_ASSERT(cacheLineSize >= 1);
    auto biasF = Vec8::load(bias);
    auto minF = Vec8(parameter[2]);
    auto maxF = Vec8(parameter[3]);
    auto SRC_TILE_UNIT = 4 * PACK_UNIT;
    auto DST_TILE_UNIT = 2 * PACK_UNIT;
    for (int x = 0; x < unit; ++x) {
        auto offset = SRC_TILE_UNIT * x;
        int i = 0;
        Vec8 m0     = Vec8::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 0) * Vec8::load(cacheLine[i] + offset + PACK_UNIT * 0);
        Vec8 m1     = Vec8::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 1) * Vec8::load(cacheLine[i] + offset + PACK_UNIT * 1);
        Vec8 m2     = Vec8::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 2) * Vec8::load(cacheLine[i] + offset + PACK_UNIT * 2);
        Vec8 m3     = Vec8::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 3) * Vec8::load(cacheLine[i] + offset + PACK_UNIT * 3);

        for (i = 1; i < cacheLineSize; ++i) {
            m0 = m0 + Vec8::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 0) * Vec8::load(cacheLine[i] + offset + PACK_UNIT * 0);
            m1 = m1 + Vec8::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 1) * Vec8::load(cacheLine[i] + offset + PACK_UNIT * 1);
            m2 = m2 + Vec8::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 2) * Vec8::load(cacheLine[i] + offset + PACK_UNIT * 2);
            m3 = m3 + Vec8::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 3) * Vec8::load(cacheLine[i] + offset + PACK_UNIT * 3);
        }
        auto o0 = m0 + m1 + m2 + biasF;
        auto o1 = m1 - m2 + m3 + biasF;
        o0 = Vec8::min(maxF, o0);
        o1 = Vec8::min(maxF, o1);
        o0 = Vec8::max(minF, o0);
        o1 = Vec8::max(minF, o1);

        Vec8::save(dest + DST_TILE_UNIT * x + 0 * PACK_UNIT, o0);
        Vec8::save(dest + DST_TILE_UNIT * x + 1 * PACK_UNIT, o1);
    }
    if (unit * 2 < ow) {
        auto offset = SRC_TILE_UNIT * unit;
        int i = 0;
        Vec8 m0     = Vec8::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 0) * Vec8::load(cacheLine[i] + offset + PACK_UNIT * 0);
        Vec8 m1     = Vec8::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 1) * Vec8::load(cacheLine[i] + offset + PACK_UNIT * 1);
        Vec8 m2     = Vec8::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 2) * Vec8::load(cacheLine[i] + offset + PACK_UNIT * 2);

        for (i = 1; i < cacheLineSize; ++i) {
            m0 = m0 + Vec8::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 0) * Vec8::load(cacheLine[i] + offset + PACK_UNIT * 0);
            m1 = m1 + Vec8::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 1) * Vec8::load(cacheLine[i] + offset + PACK_UNIT * 1);
            m2 = m2 + Vec8::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 2) * Vec8::load(cacheLine[i] + offset + PACK_UNIT * 2);
        }
        auto o0 = m0 + m1 + m2 + biasF;
        o0 = Vec8::min(maxF, o0);
        o0 = Vec8::max(minF, o0);
        Vec8::save(dest + DST_TILE_UNIT * unit, o0);
    }
}
static void _AVX_MNNConvDwF23SourceTransUnit(const float *source, float *dest, size_t unit) {
    if (unit <= 0) {
        return;
    }
    Vec8 v0 = Vec8::load(source + PACK_UNIT * 0);
    Vec8 v1 = Vec8::load(source + PACK_UNIT * 1);
    Vec8 v2;
    Vec8 v3;
    source += 2 * PACK_UNIT;

    for (int x = 0; x < unit; ++x) {
        v2 = Vec8::load(source + 0 * PACK_UNIT);
        v3 = Vec8::load(source + 1 * PACK_UNIT);
        auto m0 = v0 - v2;
        auto m1 = v1 + v2;
        auto m2 = v2 - v1;
        auto m3 = v3 - v1;

        Vec8::save(dest + PACK_UNIT * 0, m0);
        Vec8::save(dest + PACK_UNIT * 1, m1);
        Vec8::save(dest + PACK_UNIT * 2, m2);
        Vec8::save(dest + PACK_UNIT * 3, m3);

        source += (2 * PACK_UNIT);
        dest += (4 * PACK_UNIT);

        v0 = v2;
        v1 = v3;
    }
}

void _AVX_MNNSourceTransformCommonF23(const float *source, float *dest, int unit, int iw, int pad, int su, int eu) {
    for (int x = 0; x < su; ++x) {
        auto dstX = dest + 4 * PACK_UNIT * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec8 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec8::load(source + 8 * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec8::save(dstX + PACK_UNIT * 0, m0);
        Vec8::save(dstX + PACK_UNIT * 1, m1);
        Vec8::save(dstX + PACK_UNIT * 2, m2);
        Vec8::save(dstX + PACK_UNIT * 3, m3);
    }
    _AVX_MNNConvDwF23SourceTransUnit(source + PACK_UNIT * (su * 2 - pad), dest + PACK_UNIT * 4 * su, eu - su);

    for (int x = eu; x < unit; ++x) {
        auto dstX = dest + PACK_UNIT * 4 * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec8 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec8::load(source + PACK_UNIT * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec8::save(dstX + PACK_UNIT * 0, m0);
        Vec8::save(dstX + PACK_UNIT * 1, m1);
        Vec8::save(dstX + PACK_UNIT * 2, m2);
        Vec8::save(dstX + PACK_UNIT * 3, m3);
    }
}

void _AVX_MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow, const float* bias, const float* parameter) {
    int unit = ow / 2;
    auto SRC_TILE_UNIT = 4 * PACK_UNIT;
    auto DST_TILE_UNIT = 2 * PACK_UNIT;

    auto w00 = Vec8::load(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 0);
    auto w01 = Vec8::load(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 1);
    auto w02 = Vec8::load(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 2);
    auto w03 = Vec8::load(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 3);
    auto w10 = Vec8::load(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 0);
    auto w11 = Vec8::load(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 1);
    auto w12 = Vec8::load(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 2);
    auto w13 = Vec8::load(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 3);
    auto w20 = Vec8::load(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 0);
    auto w21 = Vec8::load(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 1);
    auto w22 = Vec8::load(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 2);
    auto w23 = Vec8::load(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 3);
    auto biasF = Vec8::load(bias);
    auto minF = Vec8(parameter[2]);
    auto maxF = Vec8(parameter[3]);

    for (int x = 0; x < unit; ++x) {
        auto offset = PACK_UNIT * 4 * x;
        int i = 0;
        Vec8 m0     = w00 * Vec8::load(cacheLine[0] + offset + PACK_UNIT * 0);
        Vec8 m1     = w01 * Vec8::load(cacheLine[0] + offset + PACK_UNIT * 1);
        Vec8 m2     = w02 * Vec8::load(cacheLine[0] + offset + PACK_UNIT * 2);
        Vec8 m3     = w03 * Vec8::load(cacheLine[0] + offset + PACK_UNIT * 3);

        m0 = m0 + w10 * Vec8::load(cacheLine[1] + offset + PACK_UNIT * 0);
        m1 = m1 + w11 * Vec8::load(cacheLine[1] + offset + PACK_UNIT * 1);
        m2 = m2 + w12 * Vec8::load(cacheLine[1] + offset + PACK_UNIT * 2);
        m3 = m3 + w13 * Vec8::load(cacheLine[1] + offset + PACK_UNIT * 3);

        m0 = m0 + w20 * Vec8::load(cacheLine[2] + offset + PACK_UNIT * 0);
        m1 = m1 + w21 * Vec8::load(cacheLine[2] + offset + PACK_UNIT * 1);
        m2 = m2 + w22 * Vec8::load(cacheLine[2] + offset + PACK_UNIT * 2);
        m3 = m3 + w23 * Vec8::load(cacheLine[2] + offset + PACK_UNIT * 3);

        auto o0 = m0 + m1 + m2 + biasF;
        auto o1 = m1 - m2 + m3 + biasF;
        o0 = Vec8::min(maxF, o0);
        o1 = Vec8::min(maxF, o1);
        o0 = Vec8::max(minF, o0);
        o1 = Vec8::max(minF, o1);
        Vec8::save(dest + DST_TILE_UNIT * x + 0 * PACK_UNIT, o0);
        Vec8::save(dest + DST_TILE_UNIT * x + 1 * PACK_UNIT, o1);
    }
    if (unit * 2 < ow) {
        auto offset = PACK_UNIT * 4 * unit;
        Vec8 m0     = w00 * Vec8::load(cacheLine[0] + offset + PACK_UNIT * 0);
        Vec8 m1     = w01 * Vec8::load(cacheLine[0] + offset + PACK_UNIT * 1);
        Vec8 m2     = w02 * Vec8::load(cacheLine[0] + offset + PACK_UNIT * 2);

        m0 = m0 + w10 * Vec8::load(cacheLine[1] + offset + PACK_UNIT * 0);
        m1 = m1 + w11 * Vec8::load(cacheLine[1] + offset + PACK_UNIT * 1);
        m2 = m2 + w12 * Vec8::load(cacheLine[1] + offset + PACK_UNIT * 2);

        m0 = m0 + w20 * Vec8::load(cacheLine[2] + offset + PACK_UNIT * 0);
        m1 = m1 + w21 * Vec8::load(cacheLine[2] + offset + PACK_UNIT * 1);
        m2 = m2 + w22 * Vec8::load(cacheLine[2] + offset + PACK_UNIT * 2);
        auto o0 = m0 + m1 + m2 + biasF;
        o0 = Vec8::min(maxF, o0);
        o0 = Vec8::max(minF, o0);
        Vec8::save(dest + DST_TILE_UNIT * unit, o0);
    }
}

void _AVX_ExtraInit(void* functions) {
    auto coreFunction = static_cast<MNN::CoreFunctions*>(functions);
    coreFunction->MNNSelectBlitFunction = _selectBlit;
    coreFunction->MNNPoolingAvg = (decltype(coreFunction->MNNPoolingAvg))(MNN::poolingAvg<float, Vec8, 8>);
    // Set min value as 1 << 24
    coreFunction->MNNPoolingMax = (decltype(coreFunction->MNNPoolingMax))(MNN::poolingMax<float, Vec8, 8, -16777216>);
    coreFunction->MNNSelectBinaryFunctionForFloat = _AVX2_MNNSelectBinaryFunctionForFloat;
    coreFunction->MNNCopyC4WithStride = _AVX_MNNCopyC4WithStride;
    coreFunction->MNNAddC4WithStride = _AVX_MNNAddC4WithStride;
    coreFunction->MNNScaleAndAddBias = _AVX_MNNScaleAndAddBias;
    coreFunction->MNNMatrixAdd          = _AVX_MNNMatrixAdd;
    coreFunction->MNNMatrixSub          = _AVX_MNNMatrixSub;

    coreFunction->MNNConvRunForUnitDepthWise = _AVX_MNNConvRunForUnitDepthWise;
    coreFunction->MNNConvRunForLineDepthwise = _AVX_MNNConvRunForLineDepthwise;
    coreFunction->MNNAxByClampBroadcastUnit = _AVX_MNNAxByClampBroadcastUnit;
    coreFunction->MNNStrassenMergeCFunction = _AVX_MNNStrassenMergeCFunction;
    coreFunction->MNNMultiAndDestTransformCommon23 = _AVX_MNNMultiAndDestTransformCommon23;
    coreFunction->MNNSourceTransformCommonF23 = _AVX_MNNSourceTransformCommonF23;
    coreFunction->MNNConvDwF23MulTransUnit = _AVX_MNNConvDwF23MulTransUnit;
    coreFunction->MNNReluWithSlopeChannel = _AVX_MNNReluWithSlopeChannel;
    coreFunction->MNNDeconvRunForLineDepthwise = _AVX_MNNDeconvRunForLineDepthwise;
    coreFunction->MNNDeconvRunForUnitDepthWise = _AVX_MNNDeconvRunForUnitDepthWise;
    coreFunction->MNNGridSampleInterp = _AVX_MNNGridSampleInterp;

    // sparse conv funcs
    coreFunction->MNNGetSparseMatMulPackMode = _AVX_MNNGetSparseMatMulPackMode;
    coreFunction->MNNPackedSparseMatMulEpx1 = _AVX_MNNPackedSparseMatMulEpx1EFMA;
    coreFunction->MNNPackedSparseMatMulEpx4 = _AVX_MNNPackedSparseMatMulEpx4EFMA;
}
