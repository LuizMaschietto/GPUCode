
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

// Verifica erros da API CUDA.
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "Erro CUDA: " << cudaGetErrorString(err)      \
                  << " (linha " << __LINE__ << ")\n";              \
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)

 
bool lerMatriz(const char* nomeArquivo,
               std::vector<float>& matriz,
               int& N) {
    std::ifstream arquivo(nomeArquivo);

    if (!arquivo.is_open()) {
        std::cerr << "Erro: nao foi possivel abrir "
                  << nomeArquivo << '\n';
        return false;
    }

    matriz.clear();

    float valor;
    while (arquivo >> valor) {
        matriz.push_back(valor);
    }

    if (!arquivo.eof() || matriz.empty()) {
        std::cerr << "Erro: dados invalidos em "
                  << nomeArquivo << '\n';
        return false;
    }

    // Descobre N a partir da quantidade de elementos.
    std::size_t total = matriz.size();
    N = static_cast<int>(std::sqrt(
        static_cast<double>(total)));

    if (static_cast<std::size_t>(N) * N != total) {
        std::cerr << "Erro: " << nomeArquivo
                  << " nao contem uma matriz quadrada.\n";
        return false;
    }

    return true;
}

__global__ void matmul(const float* A,
                       const float* B,
                       float* C,
                       int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

bool salvarMatriz(const char* nomeArquivo,
                  const std::vector<float>& C,
                  int N) {
    std::ofstream arquivo(nomeArquivo);

    if (!arquivo.is_open()) {
        std::cerr << "Erro: nao foi possivel criar "
                  << nomeArquivo << '\n';
        return false;
    }

    arquivo << std::fixed << std::setprecision(0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            arquivo << C[static_cast<std::size_t>(i) * N + j];

            if (j < N - 1) {
                arquivo << ' ';
            }
        }
        arquivo << '\n';
    }

    return static_cast<bool>(arquivo);
}

int main() {
    std::vector<float> A, B, C;
    int NA = 0, NB = 0;

    if (!lerMatriz("A.txt", A, NA) ||
        !lerMatriz("B.txt", B, NB)) {
        return EXIT_FAILURE;
    }

    if (NA != NB) {
        std::cerr << "Erro: A.txt e B.txt possuem "
                  << "dimensoes diferentes.\n";
        return EXIT_FAILURE;
    }

    const int N = NA;
    const std::size_t bytes = A.size() * sizeof(float);

    C.resize(A.size());

    std::cout << "Matrizes carregadas: "
              << N << " x " << N << '\n';

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_A, A.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    matmul<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(
        C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    if (!salvarMatriz("C_cuda.txt", C, N)) {
        return EXIT_FAILURE;
    }

    std::cout << "Multiplicacao concluida.\n";
    std::cout << "Tempo do kernel: "
              << elapsed_ms << " ms\n";
    std::cout << "Resultado salvo em C_cuda.txt\n";


    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}
