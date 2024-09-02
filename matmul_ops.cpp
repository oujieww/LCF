#include <torch/extension.h>
#include <omp.h>
// #include <immintrin.h>

// void gemm_avx(float* A, float* B, float* C, int M, int N, int K) {
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             __m256 sum = _mm256_setzero_ps();
//             for (int k = 0; k < K; k += 8) {
//                 __m256 a = _mm256_loadu_ps(&A[i * K + k]);
//                 __m256 b = _mm256_loadu_ps(&B[k * N + j]);
//                 sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
//             }
//             C[i * N + j] = _mm256_reduce_add_ps(sum);
//         }
//     }
// }
void gemm_transposed(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}


// GEMM函数,用于计算两个矩阵的乘积
void gemm(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

torch::Tensor matmul_cpp_parallel(torch::Tensor x, torch::Tensor y) {
    // 获取当前系统支持的最大线程数
    int max_threads = omp_get_max_threads();
    // 打印最大线程数
    // std::cout << "Maximum number of threads: " << max_threads << std::endl;
    // 设置OpenMP使用的线程数为最大线程数
    omp_set_num_threads(max_threads);
    
    
    // 检查输入张量的维度是否合法
    TORCH_CHECK(x.dim() == 4, "Input x must be a 4D tensor");
    TORCH_CHECK(y.dim() == 4, "Input y must be a 4D tensor");
    TORCH_CHECK(x.size(3) == y.size(2), "Dimensions mismatch for matrix multiplication");

    // 获取输入张量的维度
    int64_t b1 = x.size(0);
    int64_t b2 = x.size(1);
    int64_t m = x.size(2);
    int64_t k = x.size(3);
    int64_t n = y.size(3);

    // 创建输出张量
    auto output = torch::zeros({b1, b2, m, n}, x.options());

    // 使用OpenMP进行批量并行执行
    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (int64_t i = 0; i < b1; i++) {
            for (int64_t j = 0; j < b2; j++) {
                float* x_ptr = x[i][j].data_ptr<float>();
                float* y_ptr = y[i][j].data_ptr<float>();
                float* output_ptr = output[i][j].data_ptr<float>();
                gemm(x_ptr, y_ptr, output_ptr, m, n, k);
            }
        }
    }

    return output;
}

torch::Tensor matmul_cpp_parallel_v2(torch::Tensor x, torch::Tensor y) {
    // 获取当前系统支持的最大线程数
    int max_threads = omp_get_max_threads();
    // 打印最大线程数
    // std::cout << "Maximum number of threads: " << max_threads << std::endl;
    // 设置OpenMP使用的线程数为最大线程数
    omp_set_num_threads(max_threads);
    // y = y.transpose(3, 2);
    
    // 检查输入张量的维度是否合法
    TORCH_CHECK(x.dim() == 4, "Input x must be a 4D tensor");
    TORCH_CHECK(y.dim() == 4, "Input y must be a 4D tensor");
    TORCH_CHECK(x.size(3) == y.size(3), "Dimensions mismatch for matrix multiplication");

    // 获取输入张量的维度
    int64_t b1 = x.size(0);
    int64_t b2 = x.size(1);
    int64_t m = x.size(2);
    int64_t k = x.size(3);
    int64_t n = y.size(2);

    // 创建输出张量
    auto output = torch::zeros({b1, b2, m, n}, x.options());

    // 使用OpenMP进行批量并行执行
    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (int64_t i = 0; i < b1; i++) {
            for (int64_t j = 0; j < b2; j++) {
                float* x_ptr = x[i][j].data_ptr<float>();
                float* y_ptr = y[i][j].data_ptr<float>();
                float* output_ptr = output[i][j].data_ptr<float>();
                gemm_transposed(x_ptr, y_ptr, output_ptr, m, n, k);
            }
        }
    }

    return output;
}

// GEMM函数,用于计算两个矩阵的乘积
void gemm_gather(float* A, float* B, int64_t* D, float* C,  int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[D[i * K + k] * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
// GEMM函数,用于计算两个矩阵的乘积
void gemm_gather_transpose(float* A, float* B, int64_t* D, float* C,  int M, int N, int K, int d) {
    //A 1*16
    //B 128*2048
    //D 1*16
    
    // M =1, N=2048, K=16, d=128
    
    // int bias = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // bias = D[i * K + k];
                // sum += A[i * K + k] * B[k * N + j];
                sum += A[i * K + k] * B[j * d + D[i * K + k]];
            }
            C[i * N + j] = sum;
        }
    }
}

torch::Tensor matmul_cpp_parallel_gather(torch::Tensor x, torch::Tensor y, torch::Tensor z) {
    // 获取当前系统支持的最大线程数
    int max_threads = omp_get_max_threads();
    // 打印最大线程数
    // std::cout << "Maximum number of threads: " << max_threads << std::endl;
    // 设置OpenMP使用的线程数为最大线程数
    omp_set_num_threads(max_threads);
    
    
    // 检查输入张量的维度是否合法
    TORCH_CHECK(x.dim() == 4, "Input x must be a 4D tensor");
    TORCH_CHECK(y.dim() == 4, "Input y must be a 4D tensor");
    TORCH_CHECK(z.dim() == 4, "Input y must be a 4D tensor");
    // TORCH_CHECK(x.size(3) == y.size(2), "Dimensions mismatch for matrix multiplication");
    

    // 获取输入张量的维度
    int64_t b1 = x.size(0); //40
    int64_t b2 = x.size(1); // 32
    int64_t m = x.size(2); // 1
    int64_t k = x.size(3); // 16
    int64_t n = y.size(3); // 2048
    // std::cout<<m<<" "<<n<<" "<<k<<std::endl;

    // 创建输出张量
    auto output = torch::zeros({b1, b2, m, n}, x.options());

    // 使用OpenMP进行批量并行执行
    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (int64_t i = 0; i < b1; i++) {
            for (int64_t j = 0; j < b2; j++) {
                float* x_ptr = x[i][j].data_ptr<float>(); // 1*16
                float* y_ptr = y[i][j].data_ptr<float>(); // 128 * 2048 
                float* output_ptr = output[i][j].data_ptr<float>();
                int64_t* z_ptr = z[i][j].data_ptr<int64_t>(); // 1*16
                gemm_gather(x_ptr, y_ptr,  z_ptr, output_ptr, m, n, k);
            }
        }
    }

    return output;
}

torch::Tensor matmul_cpp_parallel_gather_v2(torch::Tensor x, torch::Tensor y, torch::Tensor z) {
    // 获取当前系统支持的最大线程数
    int max_threads = omp_get_max_threads();
    // 打印最大线程数
    // std::cout << "Maximum number of threads: " << max_threads << std::endl;
    // 设置OpenMP使用的线程数为最大线程数
    omp_set_num_threads(max_threads);
    
    
    // 检查输入张量的维度是否合法
    TORCH_CHECK(x.dim() == 4, "Input x must be a 4D tensor");
    TORCH_CHECK(y.dim() == 4, "Input y must be a 4D tensor");
    TORCH_CHECK(z.dim() == 4, "Input y must be a 4D tensor");
    // TORCH_CHECK(x.size(3) == y.size(2), "Dimensions mismatch for matrix multiplication");
    

    // 获取输入张量的维度
    int64_t b1 = x.size(0); //40
    int64_t b2 = x.size(1); // 32
    int64_t m = x.size(2); // 1
    int64_t k = x.size(3); // 16
    int64_t n = y.size(2); // 2048
    int64_t d = y.size(3); // 128
    
    // std::cout<<m<<" "<<n<<" "<<k<<std::endl;

    // 创建输出张量
    auto output = torch::zeros({b1, b2, m, n}, x.options());

    // 使用OpenMP进行批量并行执行
    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (int64_t i = 0; i < b1; i++) {
            for (int64_t j = 0; j < b2; j++) {
                float* x_ptr = x[i][j].data_ptr<float>(); // 1*16
                float* y_ptr = y[i][j].data_ptr<float>(); // 128 * 2048 
                float* output_ptr = output[i][j].data_ptr<float>();
                int64_t* z_ptr = z[i][j].data_ptr<int64_t>(); // 1*16
                gemm_gather_transpose(x_ptr, y_ptr,  z_ptr, output_ptr, m, n, k, d);
            }
        }
    }

    return output;
}


torch::Tensor matmul_cpp(torch::Tensor x, torch::Tensor y) {
    // 检查输入张量的维度是否合法
    TORCH_CHECK(x.dim() == 4, "Input x must be a 4D tensor");
    TORCH_CHECK(y.dim() == 4, "Input y must be a 4D tensor");
    TORCH_CHECK(x.size(3) == y.size(2), "Dimensions mismatch for matrix multiplication");

    // 获取输入张量的维度
    int64_t b1 = x.size(0);
    int64_t b2 = x.size(1);
    int64_t m = x.size(2);
    int64_t k = x.size(3);
    int64_t n = y.size(3);

    // 创建输出张量
    auto output = torch::zeros({b1, b2, m, n}, x.options());

    // 执行批量矩阵乘法
    for (int64_t i = 0; i < b1; i++) {
        for (int64_t j = 0; j < b2; j++) {
            for (int64_t p = 0; p < m; p++) {
                for (int64_t q = 0; q < n; q++) {
                    for (int64_t r = 0; r < k; r++) {
                        output[i][j][p][q] += x[i][j][p][r] * y[i][j][r][q];
                    }
                }
            }
        }
    }

    return output;
}

torch::Tensor matmul_torch(torch::Tensor x, torch::Tensor y) {
    return torch::matmul(x, y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cpp_parallel_gather", &matmul_cpp_parallel_gather, "Matrix multiplication for 4D tensors (C++ implementation)");
    m.def("matmul_cpp_parallel_gather_v2", &matmul_cpp_parallel_gather_v2, "Matrix multiplication for 4D tensors (C++ implementation)");
    m.def("matmul_cpp_parallel", &matmul_cpp_parallel, "Matrix multiplication for 4D tensors (C++ implementation)");
    m.def("matmul_cpp_parallel_v2", &matmul_cpp_parallel_v2, "Matrix multiplication for 4D tensors (C++ implementation)");
    m.def("matmul_cpp", &matmul_cpp, "Matrix multiplication for 4D tensors (C++ implementation)");
    m.def("matmul_torch", &matmul_torch, "Matrix multiplication using torch.matmul");
}