#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

using namespace thrust::placeholders;

void testSegmentedReduction() {
    //thrust::inclusive_scan_by_key

    const int N = 1000;  // sequences
    const int K = 100;   // length of sequence
    thrust::device_vector<float> data(N * K, 1);
    thrust::device_vector<float> sums(N * K);

    // convert cuda device ptr to thrust::device_vector<float>
    // raw pointer to device memory
/*
    int* raw_ptr;
    cudaMalloc((void**)&raw_ptr, N * sizeof(int));

    // wrap raw pointer with a device_ptr
    thrust::device_ptr<int> dev_ptr(raw_ptr);

    // copy memory to a new device_vector (which automatically allocates memory)
    thrust::device_vector<int> vec(dev_ptr, dev_ptr + N);

     // free user-allocated memory
    cudaFree(raw_ptr);
*/

/* // reduction
    thrust::reduce_by_key(thrust::device,
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1 / K),
        thrust::make_transform_iterator(thrust::counting_iterator<int>(N * K), _1 / K),
        data.begin(),
        thrust::discard_iterator<int>(),
        sums.begin()
    );
*/
//segmented reverse scan
    thrust::exclusive_scan_by_key(
        thrust::device,
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1 / K),
        thrust::make_transform_iterator(thrust::counting_iterator<int>(N * K), _1 / K),
        data.begin(),
        sums.begin()
    );

    // just display the first 10 results
    thrust::copy_n(sums.begin(), 1000, std::ostream_iterator<float>(std::cout, ","));
    std::cout << std::endl;
}

void testSegmentedScan() {
    //thrust::inclusive_scan_by_key

    const int N = 1000;  // sequences
    const int K = 100;   // length of sequence
    thrust::device_vector<float> data(N * K, 1);
    thrust::device_vector<float> sums(N * K);

    // transform
    // https://docs.nvidia.com/cuda/thrust/index.html

    // convert cuda device ptr to thrust::device_vector<float>
    // raw pointer to device memory
/*
    int* raw_ptr;
    cudaMalloc((void**)&raw_ptr, N * sizeof(int));

    // wrap raw pointer with a device_ptr
    thrust::device_ptr<int> dev_ptr(raw_ptr);

    // copy memory to a new device_vector (which automatically allocates memory)
    thrust::device_vector<int> vec(dev_ptr, dev_ptr + N);

     // free user-allocated memory
    cudaFree(raw_ptr);
*/

/* // reduction
    thrust::reduce_by_key(thrust::device, 
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1 / K), 
        thrust::make_transform_iterator(thrust::counting_iterator<int>(N * K), _1 / K), 
        data.begin(), 
        thrust::discard_iterator<int>(), 
        sums.begin()
    );
*/
    //segmented reverse scan
    // is important passing the right keys values to run the reversed prefix_sum
    thrust::exclusive_scan_by_key(
        thrust::device, 
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1 / K),
        thrust::make_transform_iterator(thrust::counting_iterator<int>(N * K), _1 / K),
        data.rbegin(),
        sums.rbegin()
    );

    // just display the first 10 results
    thrust::copy_n(sums.begin(), 1000, std::ostream_iterator<float>(std::cout, ","));
    std::cout << std::endl;
}