// #define FETCH_PER_WI  16

#define FETCH4(sum, id, A, jump) sum+=A[id];id+=jump;sum+=A[id];id+=jump;sum+=A[id];id+=jump;sum+=A[id];id+=jump;
#define FETCH16(sum, id, A, jump) FETCH4(sum, id, A, jump);FETCH4(sum, id, A, jump);FETCH4(sum, id, A, jump);FETCH4(sum, id, A, jump)

__kernel void bandwidth_float1(const __global float *A, __global float *B){
     int id = (get_group_id(0) * get_local_size(0) * FETCH_PER_WI) + get_local_id(0);
     float sum;
     int ls = get_local_size(0);
     for (int i = 0; i < FETCH_PER_WI/16; i++){
        // sum += A[id + i * ls];
        FETCH16(sum, id, A, ls);
     }
     B[get_global_id(0)] = sum;
}

__kernel void bandwidth_float2(const __global float2 *A, __global float *B){
     int id = (get_group_id(0) * get_local_size(0) * FETCH_PER_WI) + get_local_id(0);
     float2 sum;
     int ls = get_local_size(0);
     for (int i = 0; i < FETCH_PER_WI/16; i++){
        FETCH16(sum, id, A, ls);
     }
     B[get_global_id(0)] = sum.s0 + sum.s1;
}

__kernel void bandwidth_float4(const __global float4 *A, __global float *B){
     int id = (get_group_id(0) * get_local_size(0) * FETCH_PER_WI) + get_local_id(0);
     float4 sum;
     int ls = get_local_size(0);
     for (int i = 0; i < FETCH_PER_WI/16; i++){
        FETCH16(sum, id, A, ls);
     }
     B[get_global_id(0)] = sum.s0 + sum.s1 + sum.s2 + sum.s3;
}

__kernel void bandwidth_float8(const __global float8 *A, __global float *B){
     int id = (get_group_id(0) * get_local_size(0) * FETCH_PER_WI) + get_local_id(0);
     float8 sum;
     int ls = get_local_size(0);
     for (int i = 0; i < FETCH_PER_WI/16; i++){
        FETCH16(sum, id, A, ls);
     }
     B[get_global_id(0)] = sum.s0 + sum.s1 + sum.s2 + sum.s3 + sum.s4 + sum.s5 + sum.s6 + sum.s7;
}

__kernel void bandwidth_float16(const __global float16 *A, __global float *B){
     int id = (get_group_id(0) * get_local_size(0) * FETCH_PER_WI) + get_local_id(0);
     float16 sum;
     int ls = get_local_size(0);
     for (int i = 0; i < FETCH_PER_WI/16; i++){
        FETCH16(sum, id, A, ls);
     }
     B[get_global_id(0)] = sum.s0 + sum.s1 + sum.s2 + sum.s3 + sum.s4 + sum.s5 + sum.s6 + sum.s7 + sum.s8 + sum.s9 + sum.sa + sum.sb + sum.sc + sum.sd + sum.se + sum.sf;
}

#define MAD4(x, y) x = mad(y, x, y); y = mad(x, y, x); x = mad(y, x, y); y = mad(x, y, x);
#define MAD16(x, y) MAD4(x, y);MAD4(x, y);MAD4(x, y);MAD4(x, y);
#define MAD64(x, y) MAD16(x, y);MAD16(x, y);MAD16(x, y);MAD16(x, y);

__kernel void compute_float1(float A, __global float *B){
    float x = A;
    float y = (float)get_local_id(0);
    for(int i=0; i<WORK_PER_WI/2/64; i++){
        y = MAD64(x, y);
    }
    B[get_global_id(0)] = y;
}

__kernel void compute_float2(float A, __global float *B){
    float2 x = (float2)(A, A+1);
    float2 y = (float2)get_local_id(0);
    for(int i=0; i<WORK_PER_WI/4/64; i++){
        y = MAD64(x, y);
    }
    B[get_global_id(0)] = y.s0 + y.s1;
}

__kernel void compute_float4(float A, __global float *B){
    float4 x = (float4)(A, A+1,A+2, A+3);
    float4 y = (float4)get_local_id(0);
    for(int i=0; i<WORK_PER_WI/8/64; i++){
        y = MAD64(x, y);
    }
    B[get_global_id(0)] = y.s0 + y.s1 + y.s2 + y.s3;
}

__kernel void compute_float8(float A, __global float *B){
    float8 x = (float8)(A, A+1,A+2, A+3,A+1, A+5,A+6, A+7);
    float8 y = (float8)get_local_id(0);
    for(int i=0; i<WORK_PER_WI/16/64; i++){
        y = MAD64(x, y);
    }
    B[get_global_id(0)] = y.s0 + y.s1 + y.s2 + y.s3+y.s4 + y.s5 + y.s6 + y.s7;
}

__kernel void compute_float16(float A, __global float *B){
    float16 x = (float16)(A, A+1,A+2, A+3,A+1, A+5,A+6, A+7,A+8, A+9,A+10, A+11,A+12, A+13,A+14, A+15);
    float16 y = (float16)get_local_id(0);
    for(int i=0; i<WORK_PER_WI/32/64; i++){
        y = MAD64(x, y);
    }
    B[get_global_id(0)] = y.s0 + y.s1 + y.s2 + y.s3+y.s4 + y.s5 + y.s6 + y.s7 + y.s8 + y.s9 + y.sa + y.sb +y.sc + y.sd + y.se + y.sf;
}

//#define MAD4(x, y) x = mad(y, x, y); y = mad(x, y, x); x = mad(y, x, y); y = mad(x, y, x);
//#define MAD16(x, y) MAD4(x, y);MAD4(x, y);MAD4(x, y);MAD4(x, y);
//#define MAD64(x, y) MAD16(x, y);MAD16(x, y);MAD16(x, y);MAD16(x, y);


#define ADD(a, b, tmp) tmp=a+b;a=b;b=tmp;
#define ADD4(a, b, tmp) ADD(a, b, tmp);ADD(a, b, tmp);ADD(a, b, tmp);ADD(a, b, tmp);
#define ADD16(a, b, tmp) ADD4(a, b, tmp);ADD4(a, b, tmp);ADD4(a, b, tmp);ADD4(a, b, tmp);
#define ADD64(a, b, tmp) ADD16(a, b, tmp);ADD16(a, b, tmp);ADD16(a, b, tmp);ADD16(a, b, tmp);

__kernel void compute_float17(float A, __global float *B){
    float y = get_local_id(0);
    float x = A;
    float tmp;
    for(int i = 0; i < WORK_PER_WI /2 / 64 ;i++){
        ADD64(x,y,tmp);
    }
    B[get_global_id(0)]= x;
}