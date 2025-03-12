#include <string.h>
#include <stdio.h>
#include "rog_map_cuda/cuda_macro.cuh"

template <typename T>
class KernelDebugger{
    private: 
        int size;
        std::string kernel_name;
        cudaStream_t stream;

    public:
        int* d_status_ptr = nullptr;
        T *d_value_ptr = nullptr;

        KernelDebugger(int num_blocks, int num_size, std::string KernelName, cudaStream_t &stream){
            size = num_blocks * num_size;
            kernel_name = KernelName;
            printf("[%s] KernelDebugger Initialized with size: %d\n", kernel_name.c_str(), size);
            CHECK_ERROR(cudaMallocManaged((void**)&d_status_ptr, size * sizeof(int)));
            CHECK_ERROR(cudaMallocManaged((void**)&d_value_ptr, size * sizeof(T)));
            CHECK_ERROR(cudaStreamAttachMemAsync(stream, d_status_ptr));
            CHECK_ERROR(cudaStreamAttachMemAsync(stream, d_value_ptr));
            CHECK_ERROR(cudaMemsetAsync(d_status_ptr, 0, size * sizeof(int), stream));
            CHECK_ERROR(cudaStreamSynchronize(stream));
        }

        ~KernelDebugger(){
            cudaFree(d_status_ptr);
            cudaFree(d_value_ptr);
        }

        void checkStatus(bool print_id = false){
            int cnt[20];
            memset(cnt, 0, sizeof(cnt));
            for (int i = 0; i < size; i++){
                cnt[d_status_ptr[i]]++;
                if (d_status_ptr[i] == 0 && print_id){
                    printf("[%s] Kernel Error: %d, %d\n", kernel_name.c_str(), i, d_status_ptr[i]);
                }
            }
            for (int i = 0; i < 10; i++){
                if (cnt[i] > 0)
                    printf("[%s] Kernel Status: %d, %d\n", kernel_name.c_str(), i, cnt[i]);
            }
        }

        void checkValue(std::vector<T> &ret, int value = -1){
            for (int i = 0; i < size; i++)
                if (d_status_ptr[i] == value){
                    ret.push_back(d_value_ptr[i]);
                }
        }


        void checkValue(std::vector<T> &ret, std::vector<int> &ret_id, int value = -1){
            ret.clear();
            ret_id.clear();
            for (int i = 0; i < size; i++)
                if (d_status_ptr[i] == value){
                    ret_id.push_back(i);
                    ret.push_back(d_value_ptr[i]);
                }
        }
};