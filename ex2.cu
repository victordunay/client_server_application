#include "ex2.h"
#include <cuda/atomic>

#define N_STREAMS (64)
#define N_TB_SERIAL (1)
#define N_THREADS_Y (16)
#define N_THREADS_X (64)
#define N_THREADS_Z (1)
#define NO_EMPTY_STREAMS (-1)


/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
typedef struct 
{
    uchar *image_in;
    uchar *image_out;
    uchar *maps;
    int img_id;
} 
stream_buffers_t;

__device__ void prefix_sum(int arr[], int arr_size) {
    // TODO complete according to hw1
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

__device__
void process_image(uchar *in, uchar *out, uchar* maps) {
    // TODO complete according to hw1
}

__global__
void process_image_kernel(uchar *in, uchar *out, uchar* maps){
    process_image(in, out, maps);
}

class streams_server : public image_processing_server
{
private:
    // TODO define stream server context (memory buffers, streams, etc...)

    stream_buffers_t * stream_buffers[N_STREAMS]

    cudaStream_t streams[N_STREAMS];

    int find_available_stream(cudaStream_t * streams)
    {
        int result = NO_EMPTY_STREAMS;

        for (int streamIdx = 0; streamIdx < N_STREAMS; ++streamIdx) 
        {
				if(cudaSuccess == cudaStreamQuery(streams[streamIdx]))
                {
                    result = streamIdx;
                    break
                }
		}

        return result;
    }

    // Allocate GPU memory for a single input image and a single output image.
    stream_buffers_t *allocate_stream_buffer()
    {
        auto context = new stream_buffers_t stream_buffer;

        // allocate GPU memory for a single input image, a single output image, and maps
        CUDA_CHECK( cudaHostAlloc(&(context->image_in), IMG_WIDTH * IMG_WIDTH, 0) );
        CUDA_CHECK( cudaHostAlloc(&(context->image_out), IMG_WIDTH * IMG_WIDTH, 0) );
        CUDA_CHECK( cudaHostAlloc(&(context->maps), TILE_COUNT * TILE_COUNT * N_BINS,0) );

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
			cudaStreamCreate(&streams[streamIdx]);
            stream_buffers[streamIdx] = NULL;
		}	      
    }

    ~streams_server() override
    {
        for (int streamIdx = 0; streamIdx < N_STREAMS; ++streamIdx) 
            {
                stream_buffer_free(stream_buffers[streamIdx]);
                cudaStreamDestroy(streams[streamIdx]);
            }	
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        dim3 GRID_SIZE(N_THREADS_X, N_THREADS_Y , N_THREADS_Z);

        // TODO place memory transfers and kernel invocation in streams if possible.
        available_stream_idx = find_available_stream(&streams);

        if (NO_EMPTY_STREAMS != available_stream_idx)
        {
            stream_buffers_t *stream_buffer[available_stream_idx] = allocate_stream_buffer();
            //assign image id from client
            stream_buffer[available_stream_idx].img_id = img_id;  

            //   1. copy the relevant image from images_in to the GPU memory you allocated
            CUDA_CHECK( cudaMemcpyAsync(stream_buffer[available_stream_idx]->image_in, &img_in[image_index * IMG_WIDTH * IMG_HEIGHT], IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToDevice, streams[available_stream_idx]) );

            //   2. invoke GPU kernel on this image
            process_image_kernel<<<N_TB_SERIAL, GRID_SIZE, streams[available_stream_idx]>>>((stream_buffer[available_stream_idx]->image_in), (stream_buffer[available_stream_idx]->image_out), stream_buffer[available_stream_idx]->maps); 
            
            //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
            CUDA_CHECK( cudaMemcpyAsync(&img_out[image_index * IMG_WIDTH * IMG_HEIGHT],stream_buffer[available_stream_idx]->image_out, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToDevice, streams[available_stream_idx]) );

            return true;
        }       

        
        return false;
    }

    bool dequeue(int *img_id) override
    {
        return false;

        // TODO query (don't block) streams for any completed requests.
        //for ()
        //{
            cudaError_t status = cudaStreamQuery(0); // TODO query diffrent stream each iteration
            switch (status) {
            case cudaSuccess:
                // TODO return the img_id of the request that was completed.
                
                //*img_id = ...
                return true;
            case cudaErrorNotReady:
                return false;
            default:
                CUDA_CHECK(status);
                return false;
            }
        //}
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
