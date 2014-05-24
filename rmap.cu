#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define floattype double

#define M_PI (floattype)(3.14159265358979323846)
#define MINUS_INVERSE_2_PI (floattype)(-0.15915494309189533576888376337251)

#define SIZE 256  // part side in pixels
#define TOTAL_SIZE 256  // whole picture side in pixels
#define X 0  // part offset from left
#define Y 0  // part offset from bottom

#define THREADS 448
#define JOBS_PER_THREAD 384
#define ITER_PER_JOB 100000
#define TOTAL_JOBS (JOBS_PER_THREAD * THREADS)

#define EPS (floattype)(1e-3)
#define ITER_TAIL 1000
#define SAMPLING_TAIL 512

#define STATUS_START 0
#define STATUS_WORK 1
#define STATUS_FINISH 2
#define STATUS_UNDEF 3

struct Pixel
{
	floattype prev_sum;
	floattype sum;
	int count;
	int coord;
	int jobs_left;
	int jobs_in_progress;
	long long iterations;
	int iterations_min;
	int iterations_max;
};

struct Job
{
	floattype prev_theta;
	floattype theta;
	floattype result;
	floattype random_value;
	int count;
	int tail_count;
	int status;
	int coord;
};

__device__ floattype iterate(floattype theta, floattype omega, floattype k)
{
	return fma(k * MINUS_INVERSE_2_PI, sinpi(2*theta), theta + omega);
}

__device__ void rotation_number(
	floattype omega, 
	floattype k,
	struct Job * job)
{
	if (job->status == STATUS_FINISH || job->status == STATUS_UNDEF)
	{
		return;
	}

	if (job->status == STATUS_START)	
	{
		job->prev_theta = iterate(job->random_value, omega, k);
		job->theta = iterate(job->prev_theta, omega, k);
		job->count = 2;
		job->tail_count = 0;
		job->status = STATUS_WORK;
	}
	
	for (int i = 0; i < ITER_PER_JOB && job->tail_count < ITER_TAIL; i++)
	{
		floattype prev_result = job->prev_theta / (job->count-1);
		floattype current_result = job->theta / job->count;

		if (fabs(current_result - prev_result) < EPS)
		{
			job->tail_count++;
		}
		else
		{
			job->tail_count = 0;
		}
		
		job->prev_theta = job->theta;
		job->theta = iterate(job->theta, omega, k);
		job->count++;
	}
	
	if (job->tail_count >= ITER_TAIL)
	{
		job->status = STATUS_FINISH;
		job->result = job->theta / job->count;
	}
}

__device__ floattype get_omega(int x)
{
	return (X + x + (floattype)(0.5)) / TOTAL_SIZE;
}

__device__ floattype get_k(int y)
{
	return (Y + y + (floattype)(0.5)) / TOTAL_SIZE * 2 * M_PI;
}

__global__ void kernel(struct Job * jobs)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	struct Job * job = &jobs[i];
	
	int x = job->coord / SIZE;
	int y = job->coord % SIZE;
	
	rotation_number(get_omega(x), get_k(y), job);
}

void * my_cuda_malloc(size_t size)
{
	void * data = 0;
	cudaMalloc(&data, size);
	return data;
}

void * my_host_malloc(size_t size)
{
	void * data = 0;
	cudaMallocHost(&data, size);
	return data;
}

unsigned int * device_int(unsigned int var)
{
	unsigned int * result = (unsigned int*)my_cuda_malloc(sizeof(unsigned int));
	cudaMemcpy(result, &var, sizeof(unsigned int), cudaMemcpyHostToDevice);
	return result;
}

void report_status(int unfinished_pixels, time_t start)
{
	printf("%i\t%lf\t%lli\n", unfinished_pixels, unfinished_pixels * 100.0 / (SIZE * SIZE), (long long)(time(0) - start));
	fflush(stdout);
}

void print_error()
{
	fprintf(stderr, "%s\n", cudaGetErrorString(cudaGetLastError()));
	fflush(stderr);
}

void init_pixel(struct Pixel * pixel, int coord)
{
	pixel->prev_sum = 0.0;
	pixel->sum = 0.0;
	pixel->count = 0;
	pixel->coord = coord;
	pixel->jobs_left = SAMPLING_TAIL;
	pixel->jobs_in_progress = 0;
	pixel->iterations = 0;
	pixel->iterations_min = 0x7FFFFFFF;
	pixel->iterations_max = 0;
}

void print_pixel(struct Pixel * pixel)
{
	fprintf(stderr, "ps: %lf sm: %lf cn: %i cr: %i jl: %i jp: %i i: %lli\n", 
		pixel->prev_sum,
		pixel->sum,
		pixel->count,
		pixel->coord,
		pixel->jobs_left,
		pixel->jobs_in_progress,
		pixel->iterations);
}

struct PixelQueue
{
	struct Pixel ** pixels;
	int start;
	int count;
};

void init(struct PixelQueue * q)
{
	q->pixels = (struct Pixel**)malloc(SIZE * SIZE * sizeof(struct Pixel*));
	q->start = 0;
	q->count = 0;
}

void enqueue(struct PixelQueue * q, struct Pixel * element)
{
	q->pixels[(q->start + q->count) % (SIZE * SIZE)] = element;
	q->count++;
}

void dequeue(struct PixelQueue * q)
{
	q->start = (q->start + 1) % (SIZE * SIZE);
	q->count--;
}

struct Pixel * front(struct PixelQueue * q)
{
	return q->pixels[q->start];
}

void make_job(
	struct PixelQueue * pixels_with_work, 
	struct Job * job)
{
		if (pixels_with_work->count != 0)
		{
			struct Pixel * pixel = front(pixels_with_work);
			job->status = STATUS_START;
			job->coord = pixel->coord;
			job->random_value = rand() * 1.0 / RAND_MAX;
			pixel->jobs_left--;
			pixel->jobs_in_progress++;

			if (pixel->jobs_left == 0)
			{
				dequeue(pixels_with_work);
			}
		}
		else
		{
			job->status = STATUS_UNDEF;
		}
}

void fill_jobs(
	struct PixelQueue * pixels_with_work, 
	struct Job * host_jobs)
{
	for (int i = 0; i < TOTAL_JOBS; i++)
	{
		make_job(pixels_with_work, &host_jobs[i]);
	}
}

void gather_result(
	struct Job * job, 
	struct Pixel * pixel,
	struct PixelQueue * pixels_with_work)
{
	pixel->jobs_in_progress--;
	pixel->prev_sum = pixel->sum;
	pixel->sum += job->result;
	pixel->count++;
	pixel->iterations += job->count;
	pixel->iterations_min = (pixel->iterations_min < job->count) ? pixel->iterations_min : job->count;
	pixel->iterations_max = (pixel->iterations_min > job->count) ? pixel->iterations_min : job->count;
	
	floattype prev_result = pixel->prev_sum / (pixel->count-1);
	floattype result = pixel->sum / pixel->count;
	if (pixel->count >= 2 && fabs(result - prev_result) >= EPS)
	{
		int new_jobs_left = SAMPLING_TAIL - pixel->jobs_in_progress;
		if (pixel->jobs_left == 0 && new_jobs_left > 0)
		{
			enqueue(pixels_with_work, pixel);
		}
		pixel->jobs_left = new_jobs_left;
	}
}

void check_finish(
	struct Pixel * pixel,
	int * unfinished_pixels,
	floattype * result,
	floattype * iter,
	floattype * iter_min,
	floattype * iter_max,
	floattype * aa)
{
	if (pixel->jobs_in_progress != 0 ||
		pixel->jobs_left != 0)
	{
		return;
	}
	
	result[pixel->coord] = pixel->sum / pixel->count;
	iter[pixel->coord] = pixel->iterations / pixel->count;
	iter_min[pixel->coord] = pixel->iterations_min;
	iter_max[pixel->coord] = pixel->iterations_max;
	aa[pixel->coord] = pixel->count;
	(*unfinished_pixels)--;
}

void refill_jobs(
	int * unfinished_pixels,
	struct Pixel * pixels,
	struct PixelQueue * pixels_with_work,
	struct Job * host_jobs,
	floattype * result,
	floattype * iter,
	floattype * iter_min,
	floattype * iter_max,
	floattype * aa)
{
	for (int i = 0; i < TOTAL_JOBS; i++)
	{
		if (host_jobs[i].status != STATUS_FINISH)
		{
			continue;
		}
		
		struct Job * job = &host_jobs[i];
		struct Pixel * pixel = &pixels[job->coord];
		gather_result(job, pixel, pixels_with_work);
		check_finish(pixel, unfinished_pixels, result, iter, iter_min, iter_max, aa);
	}
	
	for (int i = 0; i < TOTAL_JOBS; i++)
	{
		if (host_jobs[i].status != STATUS_FINISH)
		{
			continue;
		}
		
		make_job(pixels_with_work, &host_jobs[i]);
	}
}

void write_binary(floattype * data, const char * filename)
{
	FILE * output = fopen(filename, "wb");
	fwrite(data, sizeof(floattype), SIZE * SIZE, output);
	fclose(output);	
}

int main()
{
	// Result malloc
	floattype * result = (floattype*)malloc(SIZE * SIZE * sizeof(floattype));
	floattype * iter = (floattype*)malloc(SIZE * SIZE * sizeof(floattype));
	floattype * iter_min = (floattype*)malloc(SIZE * SIZE * sizeof(floattype));
	floattype * iter_max = (floattype*)malloc(SIZE * SIZE * sizeof(floattype));
	floattype * aa = (floattype*)malloc(SIZE * SIZE * sizeof(floattype));
	
	// Pixels malloc
	int unfinished_pixels = SIZE * SIZE;
	struct Pixel * pixels = (struct Pixel*)malloc(SIZE * SIZE * sizeof(struct Pixel));
	struct PixelQueue pixels_with_work;
	init(&pixels_with_work);
	for (int i = 0; i < SIZE * SIZE; i++)
	{
		init_pixel(&pixels[i], i);
		enqueue(&pixels_with_work, &pixels[i]);
	}
	
	// Jobs malloc
	struct Job * host_jobs = (struct Job*)my_host_malloc(TOTAL_JOBS * sizeof(struct Job));	
	struct Job * jobs = (struct Job *)my_cuda_malloc(TOTAL_JOBS * sizeof(struct Job));
	
	time_t start = time(0);
	
	// Memory report
	long long host_memory = 
		SIZE * SIZE * (3*sizeof(floattype) + sizeof(struct Pixel) + sizeof(struct Pixel*)) + 
		TOTAL_JOBS * sizeof(struct Job);
	long long device_memory = TOTAL_JOBS * sizeof(struct Job);
	fprintf(stderr, "Total host memory: %lli bytes\n", host_memory);
	fprintf(stderr, "Total device memory: %lli bytes\n", device_memory);
			
	// Anacrusis
	printf("pixels\tpercent\ttime\n");
	fflush(stdout);
	fill_jobs(&pixels_with_work, host_jobs);
	cudaMemcpy(jobs, host_jobs, TOTAL_JOBS * sizeof(struct Job), cudaMemcpyHostToDevice);
	kernel<<< JOBS_PER_THREAD, THREADS >>>(jobs);
	report_status(unfinished_pixels, start);
	
	// Main loop
	while (unfinished_pixels > 0)
	{
		// Error
		print_error();
		// Sync
		cudaDeviceSynchronize();
		// Load
		cudaMemcpy(host_jobs, jobs, TOTAL_JOBS * sizeof(struct Job), cudaMemcpyDeviceToHost);
		// Refill
		refill_jobs(&unfinished_pixels, pixels, &pixels_with_work, host_jobs, result, iter, iter_min, iter_max, aa);
		// Store
		cudaMemcpy(jobs, host_jobs, TOTAL_JOBS * sizeof(struct Job), cudaMemcpyHostToDevice);
		// Kernel
		kernel<<< JOBS_PER_THREAD, THREADS >>>(jobs);
		// Report
		report_status(unfinished_pixels, start);
	}
	
	cudaDeviceSynchronize();
	cudaDeviceReset();
	
	// Write results
	char image_filename[256], iter_filename[256], aa_filename[256];
	sprintf(image_filename, "image_%ix%i_%i_%i.bin", TOTAL_SIZE, TOTAL_SIZE, X, Y);
	write_binary(result, image_filename);
	sprintf(iter_filename, "iter_%ix%i_%i_%i.bin", TOTAL_SIZE, TOTAL_SIZE, X, Y);
	write_binary(iter, iter_filename);
	sprintf(iter_filename, "iter_min_%ix%i_%i_%i.bin", TOTAL_SIZE, TOTAL_SIZE, X, Y);
	write_binary(iter_min, iter_filename);
	sprintf(iter_filename, "iter_max_%ix%i_%i_%i.bin", TOTAL_SIZE, TOTAL_SIZE, X, Y);
	write_binary(iter_max, iter_filename);
	sprintf(aa_filename, "aa_%ix%i_%i_%i.bin", TOTAL_SIZE, TOTAL_SIZE, X, Y);
	write_binary(aa, aa_filename);
	
	// Free
	cudaFree(jobs);
	cudaFreeHost(host_jobs);
	free(pixels);
	free(result);
	free(iter);
	free(aa);
}