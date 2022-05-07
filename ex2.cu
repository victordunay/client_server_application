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


/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
typedef struct 
{
    uchar *image_in;
    uchar *image_out;
    uchar *maps;
    int img_id;
} 
stream_buffers_t;


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
 * @param all_out Array of output images, in global memory ([N_IMAGES][IMG_HEIGHT][IMG_WIDTH])
 * @param maps 4D array ([N_IMAGES][TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @return __global__ 
 */
__device__ void process_image(uchar *in, uchar *out, uchar* maps) 
{
   __shared__ int cdf[N_BINS];
    int image_start = IMG_WIDTH * IMG_HEIGHT * blockIdx.x;
    int map_start = TILE_COUNT * TILE_COUNT * N_BINS * blockIdx.x;
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
    interpolate_device(&maps[map_start],&in[image_start], &out[image_start]);
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
                CUDA_CHECK(status);
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
        CUDA_CHECK( cudaHostAlloc(&(context->image_in), IMG_WIDTH * IMG_WIDTH, 0) );
        CUDA_CHECK( cudaHostAlloc(&(context->image_out), IMG_WIDTH * IMG_WIDTH, 0) );
        CUDA_CHECK( cudaHostAlloc(&(context->maps), TILE_COUNT * TILE_COUNT * N_BINS,0) );

        // initialize img_id 
        context->img_id = INIT_ID;

        return context;
    }

    /* Release allocated resources for the task-serial implementation. */
    void stream_buffer_free(stream_buffers_t *stream_buffer)
    {
        //TODO: free resources allocated in task_serial_init
        CUDA_CHECK(cudaFreeHost(stream_buffer->image_in));
        CUDA_CHECK(cudaFreeHost(stream_buffer->image_out));
        CUDA_CHECK(cudaFreeHost(stream_buffer->maps));
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
            (this->stream_buffers[available_stream_idx])->img_id = img_id;  
            printf("the last print works\n");
            //   1. copy the relevant image from images_in to the GPU memory you allocated
            CUDA_CHECK( cudaMemcpyAsync((this->stream_buffers[available_stream_idx])->image_in, &img_in[img_id * IMG_WIDTH * IMG_HEIGHT], IMG_WIDTH * IMG_HEIGHT, cudaMemcpyHostToDevice, this->streams[available_stream_idx]) );
            printf("the last print doesnt work\n");
            //   2. invoke GPU kernel on this image
            process_image_kernel<<<N_TB_SERIAL, GRID_SIZE, 0, streams[available_stream_idx]>>>(((this->stream_buffers[available_stream_idx])->image_in), ((this->stream_buffers[available_stream_idx])->image_out), (this->stream_buffers[available_stream_idx])->maps); 
            //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
            CUDA_CHECK( cudaMemcpyAsync(&img_out[img_id * IMG_WIDTH * IMG_HEIGHT],(this->stream_buffers[available_stream_idx])->image_out, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToHost, streams[available_stream_idx]) );
            return true;
        }       

        return false;
    }

    bool dequeue(int *img_id) override
    {
        int id = checkFinishedStreams();
        if(id == NO_EMPTY_STREAMS)
            return false;
        *img_id = id;
        return true;   
    }
};




std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

// TODO implement a lock
// TODO implement a MPMC queue
// TODO implement the persistent kernel
// TODO implement a function for calculating the threadblocks count

class queue_server : public image_processing_server
{

private:
    // TODO define queue server context (memory buffers, etc...)
public:
    queue_server(int threads)
    {
        // TODO initialize host state
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        return false;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        return false;

        // TODO return the img_id of the request that was completed.
        //*img_id = ... 
        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}

