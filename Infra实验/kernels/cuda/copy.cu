// Completed infrastructure demo only. This is NOT a solution to an exercise.
extern "C" __global__ void copy_bytes(
    const unsigned char* input, unsigned char* output, unsigned long long nbytes) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nbytes) output[i] = input[i];
}
