// Exercise only: no RMSNorm algorithm has been supplied.
// Remove this directive when you start writing your own implementation.
// #error "TODO: implement your CUDA RMSNorm kernel"

//warp_reduce_sum

//block_reduce_sum
__device__ float block_reduce_sum(float val) {
    __shared__ float s[256];
    //1. 每个thread把自己的val写进s[threadIdx.x]
    s[threadIdx.x] = val;
    //2. 让已经执行完的代码在这里等待
    __syncthreads();
    //3. thread 0 把 s[0] - s[255]加起来，结果写回s[0]
    if(threadIdx.x == 0){
        for (int i=1; i<256; i++){
            s[0] += s[i];
        }
    }
    //4. 确认thread0执行完了
    __syncthreads();
    //5. 每个thread都读s[0]作为返回值
    return s[0];
}

extern "C" __global__ void rms_norm_f32(
    const float* x, const float* weight, float* output,
    int rows, int cols, float eps) {
    // Your implementation goes here.

    int row = blockIdx.x;
    const float* x_row = x + (size_t)row*cols;
    float* y_row = output + (size_t)row*cols;

    float sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x){
        float v = *(x_row + j);
        sum += v*v;
    }
    sum = block_reduce_sum(sum);

    float mean = sum/cols;
    float inv_rms = rsqrtf(mean + eps);

    for (int j = threadIdx.x; j < cols; j += blockDim.x){
        *(y_row + j) = *(x_row + j) * inv_rms * *(weight + j);
    }
}
