//
//  Arm82OptFunc.hpp
//  MNN
//
//  Created by MNN on 2019/02/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#ifndef Arm82OptFunc_hpp
#define Arm82OptFunc_hpp

#include "Arm82Backend.hpp"
#include "core/Macro.h"

void Arm82MNNGetMatMulPackMode(int* eP, int *lP, int* hP);
void Arm82MNNExp(FLOAT16* dst, const FLOAT16* src, size_t dataSize);
void MNNQuantizeFP16(const float* src, int16_t* dst, size_t size);
void MNNDequantizeFP16(const int16_t* src, float* dst, size_t size);
void MNNPackC8FP16(FLOAT16* dest, const FLOAT16* source, size_t area, size_t depth);
void MNNUnPackC8FP16(FLOAT16* dest, const FLOAT16* source, size_t area, size_t depth);
// nc4hw4 to nc8hw8(aka fp32 -> fp16), convete dataformat and data type
void MNNNC4HW4TONC8HW8(FLOAT16* dest, const float* source, size_t plane, size_t channel);
// nc8hw8 to nc4hw4(aka fp16 -> fp32)
void MNNNC8HW8TONC4HW4(float* dest, const FLOAT16* source, size_t plane, size_t channel);

template <typename TIN, typename TOUT, int UNIT>
void MNNPackUNIT(TOUT* dst, const TIN* src, size_t area, size_t depth) {
    int depthCUnit  = depth / UNIT;
    int depthRemain = depthCUnit * UNIT;
    int remain      = depth - depthRemain;
    int z, x, y;
    const TIN* srcChannel[UNIT];
    const TIN* srcOffset = src;
    for(z = 0; z < depthCUnit; ++z) {
        for(y = 0; y < UNIT; ++y) {
            srcChannel[y] = srcOffset + area * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < UNIT; ++y) {
                dst[0] = TOUT(srcChannel[y][0]);
                srcChannel[y]++;
                dst++;
            }
        }
        srcOffset += area * UNIT;
    }
    if(remain > 0){
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + area * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < remain; ++y) {
                dst[0] = TOUT(srcChannel[y][0]);
                srcChannel[y]++;
                dst++;
            }
            for(y = remain; y < UNIT; ++y) {
                dst[0] = 0;
                dst++;
            }
        }
    }
}

template <typename TIN, typename TOUT, int UNIT>
void MNNUnpackUNIT(TOUT* dst, const TIN* src, size_t area, size_t depth) {
    int depthCUnit  = depth / UNIT;
    int depthRemain = depthCUnit * UNIT;
    int remain      = depth - depthRemain;
    int z, x, y;
    const TIN* srcChannel[UNIT];
    const TIN* srcOffset = src;
    for(z = 0; z < depthCUnit; ++z) {
        for(y = 0; y < UNIT; ++y) {
            srcChannel[y] = srcOffset + y;
            for(x = 0; x < area; ++x) {
                dst[0] = TOUT(srcChannel[y][0]);
                srcChannel[y] += UNIT;
                dst++;
            }
        }
        srcOffset += area * UNIT;
    }
    if(remain > 0){
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + y;
            for(x = 0; x < area; ++x) {
                dst[0] = TOUT(srcChannel[y][0]);
                srcChannel[y] += UNIT;
                dst++;
            }
        }
    }
}

template<typename T, typename U>
void MNNSlowCopy(T* dst, const U* src, size_t size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = (T)src[i];
    }
}

#endif // Arm82OptFunc_hpp
#endif
