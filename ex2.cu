#include "ex2.h"
#include <cuda/atomic>

#define N_STREAMS (64)
#define N_BINS (256)
#define N_TB_SERIAL (1)
#define N_THREADS_Y (16)
#define N_THREADS_X (64)
#define N_THREADS_Z (1)
#define NO_EMPTY_STREAMS (-1)
#define INIT_ID (-1)
#define SHARED_MEM_PER_BLOCK (2048)
#define REGISTERS_PER_THREAD (32)
#define DEVICE (0)

typedef cuda::atomic<bool> atomic_lock_t;


//using namespace cuda;

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
typedef struct 
{
    uchar *image_in;
    uchar *image_out;
    uchar *maps;
    int img_id;
} 
stream_buffers_t;

typedef struct 
{
    uchar *image_in;
    uchar *image_out;
    uchar *maps;
    uchar *cpu_img_out;
    int img_id;
} 
queue_buffers_t;



enum device_t {CPU = 0, GPU};

/*************************************************************************************************/
/*                                          image proccesing aux                                 */
/*************************************************************************************************/


 /**
  * @brief Create a histogram of the tile pixels. Assumes that the kernel runs with more than 256 threads
  * 
  * @param image_start The start index of the image the block processing
  * @param t_row  The tile's row
  * @param t_col  The tile's column
  * @param histogram - the histogram of the tile
  * @param image - The images array to process.
  */
 __device__ void create_histogram(int image_start, int t_row, int t_col ,int *histogram, uchar *image)
 {
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // initialize histogram
    if(tid < N_BINS)
    {
        histogram[tid] = 0;
    }
    __syncthreads();
    //calculates the pixel index that assigned to the thread 
    int row_base_offset = (t_row * TILE_WIDTH + threadIdx.y) * IMG_WIDTH ;
    int row_interval = N_THREADS_Y * IMG_WIDTH;
    int col_offset = t_col * TILE_WIDTH + threadIdx.x; 

    uchar pixel_value = 0;

    //The block has 16 rows, Therefore, it runs 4 times so every warp run on 4 different rows
    for(int i = 0; i < TILE_WIDTH/N_THREADS_Y; i++ ) 
    {
        pixel_value = image[image_start + row_base_offset + (i * row_interval) + col_offset];
        atomicAdd(&(histogram[pixel_value]), 1);
    } 
 }

 /**
  * @brief Calculates inclusive prefix sum of the given array. Saves the sum in the given array.
  *      Assumes n_threads > arr_size
  * 
  * @param arr The given array 
  * @param arr_size The size of the array
  */
__device__ void prefix_sum(int arr[], int arr_size) 
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int increment;
    for (int stride = 1; stride < arr_size; stride *= 2) 
    {
        if (tid >= stride && tid < arr_size) 
                increment = arr[tid - stride];
        __syncthreads(); 
        if (tid >= stride && tid < arr_size) 
                arr[tid] += increment;
        __syncthreads();
    }
    return;
}

/**
 * @brief Calculates a map from the cdf and saves it in the given index in the 'maps' array.
 * 
 * @param map_start The start index in the 'maps' array of the current image's map
 * @param t_row The tile's row
 * @param t_col The tile's column
 * @param cdf The cdf of the tile.
 * @param maps Array of the maps of all images
 * @return __device__ 
 */
__device__ void calculate_maps(int map_start, int t_row, int t_col, int *cdf, uchar *maps)
{
    uchar div_result = (uchar) 0;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < N_BINS)
    {
        div_result = (uchar)(cdf[tid] * 255.0/(TILE_WIDTH*TILE_WIDTH));
        maps[map_start + (t_col + t_row * TILE_COUNT)*N_BINS + tid] = div_result;
    }   
    __syncthreads();     
}
/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__
 void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);


/**
 * @brief process an image which assigned to the block index. It takes an image given in all_in, and return the processed image in all_out respectively.
 * 
 * @param in Array of input images, in global memory ([N_IMAGES][IMG_HEIGHT][IMG_WIDTH])
 * @param out Array of output images, in global memory ([N_IMAGES][IMG_HEIGHT][IMG_WIDTH])
 * @param maps 4D array ([N_IMAGES][TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @return __global__ 
 */
__device__ void process_image(uchar *in, uchar *out, uchar* maps) 
{
   __shared__ int cdf[N_BINS];
    //int image_start = IMG_WIDTH * IMG_HEIGHT * blockIdx.x;
    //int map_start = TILE_COUNT * TILE_COUNT * N_BINS * blockIdx.x;
    for(int t_row = 0; t_row< TILE_COUNT; ++t_row)
    {
        for(int t_col = 0; t_col< TILE_COUNT; ++t_col)
        {
            create_histogram(image_start,t_row, t_col, cdf, in);
            __syncthreads();
            prefix_sum(cdf, N_BINS);
            calculate_maps(map_start, t_row, t_col,cdf, maps); 
            __syncthreads();
        }
    }
    interpolate_device(&maps,&in, &out);
    return; 

}


/*********************************************************************************************************/
/*                                      streams_server class                                             */
/*********************************************************************************************************/
__global__
void process_image_kernel(uchar *in, uchar *out, uchar* maps){
    process_image(in, out, maps);
}

class streams_server : public image_processing_server
{
private:
    // TODO define stream server context (memory buffers, streams, etc...)
    stream_buffers_t * stream_buffers[N_STREAMS];
    cudaStream_t streams[N_STREAMS];
    bool streams_availability[N_STREAMS];

   
    /**
     * @brief Checks if any of the working streams has finished. If there is, it change to a free stream and its img_id returned
     * 
     * @return int the image_id of the finished stream. return NO_EMPTY_STREAMS if there no new finish stream.
     */
    int checkFinishedStreams(void)
    {
        int result = NO_EMPTY_STREAMS;
        cudaError_t status = cudaErrorNotReady;

        for (int streamIdx = 0; streamIdx < N_STREAMS; ++streamIdx) 
        {

            if(!streams_availability[streamIdx])
            {
                status = cudaStreamQuery(this->streams[streamIdx]);
                //CUDA_CHECK(status);
                if(cudaSuccess == status)
                {
                    streams_availability[streamIdx] = true;
                    result = streamIdx;
                    break;
                }
            }
		}
        return result;
    }

    /**
     * @brief Checks for empty streams
     * 
     * @return int the stream id if empty. NO_EMPTY_STREAMS if there is no empty stream.
     */
    int findAvailableStream(void)
    {
        for (int streamIdx = 0; streamIdx < N_STREAMS; ++streamIdx) 
        {
            if(streams_availability[streamIdx])
                return streamIdx;
        }
        return NO_EMPTY_STREAMS;
    }


    // Allocate GPU memory for a single input image and a single output image.
    stream_buffers_t *allocate_stream_buffer(void)
    {
        auto context = new stream_buffers_t;

        // allocate GPU memory for a single input image, a single output image, and maps
        CUDA_CHECK( cudaMalloc(&(context->image_in), IMG_WIDTH * IMG_WIDTH) );
        CUDA_CHECK( cudaMalloc(&(context->image_out), IMG_WIDTH * IMG_WIDTH) );
        CUDA_CHECK( cudaMalloc(&(context->maps), TILE_COUNT * TILE_COUNT * N_BINS) );

        // initialize img_id 
        context->img_id = INIT_ID;

        return context;
    }

    /* Release allocated resources for the task-serial implementation. */
    void stream_buffer_free(stream_buffers_t *stream_buffer)
    {
        //TODO: free resources allocated in task_serial_init
        CUDA_CHECK(cudaFree(stream_buffer->image_in));
        CUDA_CHECK(cudaFree(stream_buffer->image_out));
        CUDA_CHECK(cudaFree(stream_buffer->maps));
        free(stream_buffer);
    }

public:

    streams_server()
    { 
		for (int streamIdx = 0; streamIdx < N_STREAMS; ++streamIdx) 
        {
			CUDA_CHECK(cudaStreamCreate(&streams[streamIdx]));
            stream_buffers[streamIdx] = allocate_stream_buffer();
            streams_availability[streamIdx] = true;
		}
        cudaDeviceSynchronize();
    }

    ~streams_server() override
    {
        for (int streamIdx = 0; streamIdx < N_STREAMS; ++streamIdx) 
        {
            stream_buffer_free(stream_buffers[streamIdx]);
            CUDA_CHECK(cudaStreamDestroy(streams[streamIdx]));
        }	
    }


    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        dim3 GRID_SIZE(N_THREADS_X, N_THREADS_Y , N_THREADS_Z);


        // TODO place memory transfers and kernel invocation in streams if possible.
        int available_stream_idx = findAvailableStream();
        if (available_stream_idx != NO_EMPTY_STREAMS)
        {
            //assign image id from client
            this->stream_buffers[available_stream_idx]->img_id = img_id;
            this->streams_availability[available_stream_idx] = false;
            //printf("%d",available_stream_idx);
            //   1. copy the relevant image from images_in to the GPU memory you allocated
            cudaMemcpyAsync(this->stream_buffers[available_stream_idx]->image_in, img_in, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyHostToDevice, this->streams[available_stream_idx]);
            //   2. invoke GPU kernel on this image
            process_image_kernel<<<N_TB_SERIAL, GRID_SIZE, 0, streams[available_stream_idx]>>>(this->stream_buffers[available_stream_idx]->image_in, this->stream_buffers[available_stream_idx]->image_out, this->stream_buffers[available_stream_idx]->maps); 

            //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
            cudaMemcpyAsync(img_out, this->stream_buffers[available_stream_idx]->image_out, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToHost, this->streams[available_stream_idx]); 

            
            return true;
        }       

        return false;
    }

    bool dequeue(int *img_id) override
    {
        // check if any job was finished , and assign the corresponding stream index to finished_stream_idx
        int finished_stream_idx = checkFinishedStreams();

        if(finished_stream_idx == NO_EMPTY_STREAMS)
            return false;

        // get the img_id of the finished job 
        *img_id = this->stream_buffers[finished_stream_idx]->img_id;
        return true;   
    }
};


std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

/*********************************************************************************************************/
/*                                      queue_server class                                             */
/*********************************************************************************************************/

// TODO implement a lock
// TODO implement a MPMC queue
// TODO implement the persistent kernel
// TODO implement a function for calculating the threadblocks count

/*
typedef struct 
{
    uchar *image_in;
    uchar *image_out;
    uchar *maps;
} 
queue_buffers_t;*/


class shared_queue 
{
private:

    //queue data
    size_t queue_size;
    //device_t device;
    
    int *data;
    uchar **in;
    uchar **out;

    //queue sync variables
    cuda::atomic<size_t> _head;
    cuda::atomic<size_t> _tail;
    atomic_lock_t _readerlock;
    atomic_lock_t _writerlock;
      


    //locks functions

    __device__  __host__ void Lock(atomic_lock_t * _lock) 
    {
        printf("before exchange\n");
        while(_lock->load(cuda::memory_order_acquire) == true);
        while(_lock->exchange(true, cuda::memory_order_acq_rel));
        printf("after exchange 2\n");
        //atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);

    }

    __device__ __host__ void Unlock(atomic_lock_t * _lock) 
    {
        _lock->store(false, cuda::memory_order_release);

    }

public:

    /**
     * @brief enqueue an image by sending the id of the image. sending -1 by the cpu is for terminate the kernel
     * 
     * @param img_id the id of the imag
     */
    __device__  __host__ void enqueue_response(int img_id)
    {
        Lock(_writerlock);
        int tail =  _tail.load(cuda::memory_order_relaxed);
        while (tail - _head.load(cuda::memory_order_acquire) == queue_size);
            //Unlock(_writerlock);
            //Lock(_writerlock);
            //tail =  _tail.load(cuda::memory_order_relaxed);
        

        //cpu copy from cpu memory to gpu memory. gpu copy from gpu memory to another gpu memory
        //cudaMemcpyKind kind = (cpu_side) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice ;
        
        data[tail % queue_size] = img_id;
        _tail.store(tail + 1, cuda::memory_order_release);
        Unlock(_writerlock);
    }

    __device__  __host__ int dequeue_request(uchar *in, uchar *out, uchar* maps)

    {
        int img_id = 0;
        Lock(&(this->_readerlock));
        int head = _head.load(cuda::memory_order_relaxed);
        while (_tail.load(cuda::memory_order_acquire) == _head)
        /*{
            Unlock(_readerlock);
            Lock(_readerlock);
            head = _head.load(cuda::memory_order_relaxed);
        }*/
        switch (this->device) 
        {
            case CPU:
                img_id = this->data[head % queue_size]->img_id;
                break;
            case GPU:
                //cpu copy from gpu memory to cpu memory. gpu copy from gpu memory to another gpu memory
                img_id = this->data[head % queue_size]->img_id;
                out = this->data[head % queue_size]->image_out;
                in = this->data[head % queue_size]->image_in;
                maps = this->data[head % queue_size]->maps;
                //int img_id = data[head % queue_size];
                break;
        }
  
        _head.store(head + 1, cuda::memory_order_release);
        Unlock(&_readerlock);
        return img_id;
    }

    // Allocate GPU memory for a single input image and a single output image.
    queue_buffers_t *allocate_queue_buffer(void)
    {
        auto context = new queue_buffers_t;

        // allocate GPU memory for a single input image, a single output image, and maps
        CUDA_CHECK( cudaMalloc(&(context->image_in), IMG_WIDTH * IMG_WIDTH) );
        CUDA_CHECK( cudaMalloc(&(context->image_out), IMG_WIDTH * IMG_WIDTH) );
        CUDA_CHECK( cudaMalloc(&(context->maps), TILE_COUNT * TILE_COUNT * N_BINS) );
        context->img_id = 0;
        context->cpu_img_out = NULL;

        return context;
    }


    shared_queue(int queue_size):queue_size(queue_size),cpu_side(cpu_side),data(nullptr),in(nullptr),out(nullptr),_head(0),_tail(0),_readerlock(false),_writerlock(false)
    {   
        // Allocate queue memory
        //size_t size_in_bytes = queue_size * sizeof(int);
        /*
        this->device = device;
        this->queue_size = queue_size;
        this->_head = 0;
        this->_tail = 0;
        Unlock(&_readerlock);
        Unlock(&_writerlock);*/

  
        size_t size_in_bytes = queue_size * sizeof(int);
        CUDA_CHECK(cudaMallocHost(&((void*)data), size_in_bytes));
        CUDA_CHECK(cudaMemset(((void*)data), 0, size_in_bytes));
        CUDA_CHECK(cudaMallocHost(&((void*)in), size_in_bytes));
        CUDA_CHECK(cudaMemset(((void*)in), 0, size_in_bytes));
        CUDA_CHECK(cudaMallocHost(&((void*)out), size_in_bytes));
        CUDA_CHECK(cudaMemset(((void*)out), 0, size_in_bytes));
    }

    ~shared_queue() 
    {
        for (int slotIdx = 0; slotIdx < queue_size; ++slotIdx) 
            {
                CUDA_CHECK(cudaFree(data[slotIdx]->image_in));
                CUDA_CHECK(cudaFree(data[slotIdx]->image_out));
                CUDA_CHECK(cudaFree(data[slotIdx]->maps));
            }	
    }
};

__global__
void consumer_proccessor(shared_queue *gpu_to_cpu_q,shared_queue *cpu_to_gpu_q, uchar* maps)
{
    __shared__ int img_id;
    __shared__ uchar *in;
    __shared__ uchar *out;

    if(threadIdx.y + threadIdx.x == 0 )
        img_id = cpu_to_gpu_q.dequeue_request(&in,&out);
    __syncthreads();
    while(img_id)
    {
        //did not finish this
        //CUDA_CHECK( cudaMemcpy(stream_buffers[available_stream_idx]->image_in, img_in, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyHostToDevice, streams[available_stream_idx]) );
        process_image(in[img_id * IMG_WIDTH * IMG_HEIGHT], out[img_id * IMG_WIDTH * IMG_HEIGHT], maps);
        if(threadIdx.y + threadIdx.x == 0 )
            gpu_to_cpu_q.enqueue_response(img_id,&in,&out);
        __syncthreads();
        if(threadIdx.y + threadIdx.x == 0 )
            img_index = cpu_to_gpu_q.dequeue_request();
        __syncthreads();
    }
}
    
}

class queue_server : public image_processing_server
{

private:
    // TODO define queue server context (memory buffers, etc...)
    shared_queue *gpu_to_cpu_q;
    shared_queue *cpu_to_gpu_q;
    int threadblocks;
    char* pinned_host_buffer;
   

    int calcNumOfTB(int threads)
    {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, DEVICE));

        int num_of_multi = prop.multiProcessorCount;
        int threads_per_multi = prop.maxThreadsPerMultiProcessor ;
        int regs_per_multi = prop.regsPerMultiprocessor;
        int shared_per_multi = prop.sharedMemPerMultiprocessor;
        
        int threads_bound = (int) (floor(threads_per_multi/threads)*num_of_multi);
        int regs_bound = (int) (floor(regs_per_multi/(REGISTERS_PER_THREAD*threads))*num_of_multi);
        int shared_mem_bound = (int) (floor(shared_per_multi/SHARED_MEM_PER_BLOCK)*num_of_multi);

        //printf("number of SMs : %d\n\n every SM supports %d threads, %d registers and %d shared memory\n",num_of_multi,threads_per_multi,regs_per_multi,shared_per_multi);
        //printf("thus the bounds are:\n %d for threads\n %d for registers\n %d for shared memory\n",threads_bound,regs_bound,shared_mem_bound);
        //int lesser = threads_bound - regs_bound;
        if(threads_bound > shared_mem_bound)
        {
            return (shared_mem_bound>regs_bound) ? regs_bound : shared_mem_bound;
        }
        else
        {
            return (threads_bound>regs_bound) ? regs_bound : threads_bound;
        }
        return -1;
    }


 

public:
    queue_server(int threads)
    {
        // TODO initialize host state
        threadblocks = calcNumOfTB(threads);
        int num_of_slots =(int) (pow(2,ceil(log2(16*threadblocks))));
        printf("%d number of slots:%d",threadblocks,num_of_slots);
        
        cudaMallocHost(&pinned_host_buffer, 2 * sizeof(shared_queue));
        // Use placement new operator to construct our class on the pinned buffer
        shared_queue *cpu_to_gpu_q = new (pinned_host_buffer) shared_queue(num_of_slots, CPU);
        shared_queue *gpu_to_cpu_q = new (pinned_host_buffer + sizeof(shared_queue)) shared_queue(num_of_slots, GPU);

        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks

        //initiate data for proccessing - allocating arrays of data like in bulk for temp use.
      
        /*CUDA_CHECK( cudaMalloc(&image_in,threadblocks * IMG_WIDTH * IMG_WIDTH ,0) );
        CUDA_CHECK( cudaMalloc(&image_out,threadblocks * IMG_WIDTH * IMG_WIDTH,0) );*/
        uchar* maps;
        CUDA_CHECK( cudaMalloc(&maps,threadblocks * TILE_COUNT * TILE_COUNT * N_BINS,0) );

         //kernel invocing
        dim3 GRID_SIZE(N_THREADS_X, threads/N_THREADS_X , N_THREADS_Z);
        consumer_proccessor<<<threadblocks, GRID_SIZE>>>(gpu_to_cpu_q,cpu_to_gpu_q,maps);
    }

    ~queue_server() override
    {
        for(int i= 0; i<threadblocks; ++i)
        {
            cpu_to_gpu_q->enqueue_response(-1, nullptr, nullptr);
        }
        cudaDeviceSynchronize();
        // TODO free resources allocated in constructor
        gpu_to_cpu_q->~shared_queue();
        cpu_to_gpu_q->~shared_queue();
        cudaFreeHost(pinned_host_buffer);
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        //check if full
            //if full return false
        cpu_to_gpu_q->enqueue_response(img_id, img_in, img_out);
        return true;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        //implement isEmpty _head _tail query
            //return false
        // TODO return the img_id of the request that was completed.
        *img_id = gpu_to_cpu_q->dequeue_request(this->image_in, this->image_out, this->maps);
        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
