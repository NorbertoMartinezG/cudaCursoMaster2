/*
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
*/


//-------------------------------------219 Understadn the device better---------------------------------------

/* computer architectures classification
		-SISD - Single instruction single data
		-SIMD - single instruction multiple data
		-MISD - multiple instruction single data
		-MIMD - multiple instruction multiple data 
	CUDA se basa en SIMD
		-SIMT - single instruction multiple threads

*/


//------------------------------------- 220 Warps ---------------------------------------

/*
* -thread blocks are divide in to smaller units called warps each having 32 consecutive threads
* -warps can be defined asthe basic unit of execution in a SM
* -all threads in a warp are executed in single instrucction multiple thread (SIMT) fashion
* 
* -Los bloques se dividen en subbloques de 32
*	-ej.1. Un bloque de 128 se dividira en 4 bloques de 32 (0-31)(32-63)(64-95)(96-127)
*	-ej.2. Un bloque de 80 se dividira en 4 bloques de los cuales se dividiran (0-31)(32-39)(40-71)(72-79)
*	-ej.3. Un bloque de 1 thread activara un bloque de 32 threads de los cuales solo utilizara 1.
*/

// EJEMPLO DE USO DE BLOQUES CUANDO SE UTILIZAN 40 HILOS
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <stdlib.h>
//
//__global__ void print_details_of_warps()
//{
//	int gid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
//
//	int warp_id = threadIdx.x / 32;
//
//	int gbid = blockIdx.y * gridDim.x + blockIdx.x;
//
//	printf( "tid: %d, bid.x: %d, bid.y: %d, gid: %d, warp_id: %d, gbid: %d \n",
//		threadIdx.x, blockIdx.x, blockIdx.y, gid, warp_id, gbid);
//}
//
//
//int main(int argc, char** argv) {
//
//	dim3 block_size(42);
//	dim3 grid_size(2, 2);
//
//	print_details_of_warps << <grid_size, block_size >> > ();
//	cudaDeviceSynchronize();
//
//	cudaDeviceReset();
//	return EXIT_SUCCESS;
//
//}

//------------------------------------- 221 Warp divergence ---------------------------------------
/*
	- forzar a algunos hilos en el warp para ejecutar diferentes instrucciones
	- la eficiencia de las ramas se puede medir para un kernel utilizando la herramienta de creacion de perfiles nvprof.

*/

//// EJEMPLO DE DIVERGENCIA
//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
//
//#include "cuda.h"
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
////#include "cuda_common.cuh"
//
//// EJECUTAR EN TERMINAL******************************
//__global__ void code_without_divergence() // asignacion de valores a A y B basados en WARP
//{
//	int gid = blockIdx.x * blockDim.x + threadIdx.x;  // id de thread global
//
//	float a, b;
//	a = b = 0;
//
//	int warp_id = gid / 32;
//
//	if (warp_id % 2 == 0)
//	{
//		a = 100.0;
//		b = 50.0;
//
//
//	}
//	else
//	{
//		a = 200;
//		b = 75;
//	}
//
//}
//
//__global__ void code_with_divergence() // asignacion de valores a A y B basados en THREAD
//{
//	int gid = blockIdx.x * blockDim.x + threadIdx.x;  // id de thread global
//
//	float a, b;
//	a = b = 0;
//
//	if (gid % 2 == 0)
//	{
//		a = 100.0;
//		b = 50.0;
//
//
//	}
//	else
//	{
//		a = 200;
//		b = 75;
//	}
//
//}
//
//
//int main(int argc, char** argv)
//{
//	printf("\n---------------------------WARP DIVERGENCE EXAMPLE----------------------\n\n");
//
//	int size = 1 << 22;
//
//	dim3 block_size(128);
//	dim3 grid_size((size + block_size.x - 1) / block_size.x);
//
//	code_without_divergence << <grid_size, block_size >> > ();
//	cudaDeviceSynchronize();
//
//	code_with_divergence << <grid_size, block_size >> > ();
//	cudaDeviceSynchronize();
//
//
//	cudaDeviceReset();
//	return 0;
//}

//------------------------------------- 222 Latency ---------------------------------------

//------------------------------------- 223 Occupancy ---------------------------------------

/*

Occupancy = Active warps / maximum warps

48 registers per thread
reg_per_warp = 48 * 32 = 1536

GTX 970 device = 65536 regs per SM

Warps permitidos por SM = 65536 / 1536 = 42.67

*/

//CUDA OCCUPANCY CALCULATOR (hoja de excel)

/* GUIDE LINE FOR GRID AND BLOCK SIZE
	-keep the number of threads per block a multiple of warp size 32
	-Avoid small block sizes: Start with at least 128 or 256 threads per block
	-keep the number of blocks much greater than the number of SMs to expose sufficient parallelism to your device

 
*/

//------------------------------------- 224 Profiling with nvprof ---------------------------------------
/*
* sm_efficient
* achieved_occupacy
*/
//------------------------------------- 226 Parallel reduction as synchronization example----------------

/*
	-cudaDeviceSynchronize -- bloquea la ejecucion de la aplicacion host hasta que las operaciones en el host esten terminadas
	- _syncthreads -- proporciona la sincronizacion en un bloque dentro del device (obliga a los hilos esperar hasta que todos los hilos lleguen a un punto

*/

//Ejemplo suma elementos de un vector  (reduccion paralela)

#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"
#include "cuda_common.cuh"


__global__ void reduction_neighbored_pairs(int * input, int * temp, int size)
{
	//identificacion del hilo
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid>size)
	{
		return;
	}

	for (int offset = 1; offset <= blockDim.x/2; offset *=2)
	{
		if (tid % (2 * offset) == 0)
		{
			input[gid] += input[gid + offset];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		temp[blockIdx.x] = input[gid];

	}

}


int main(int argc, char** argv)
{
	printf("Running neighbored pairs reduction kernel \n");

	int size = 1 << 27; // 128 Mb of data
	int byte_size = size * sizeof(int);
	int block_size = 128;

	int* h_input, * h_ref;
	h_input = (int*)malloc(byte_size);

	initialize(h_input, size, INIT_RANDOM);

	//get the reduction result from cpu
	int cpu_result = reduction_cpu(h_input, size);

	dim3 block(block_size);
	dim3 grid(size / block.x);

	printf("kernel launch parameters | grid.x : %d, block.x : %d", grid.x, block.x);

	int temp_array_byte_size = sizeof(int) * grid.x;
	h_ref = (int*)malloc(temp_array_byte_size);

	int* d_input, * d_temp;

	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));

	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size)); // establece valor inicial en 0 
	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice));

	reduction_neighbored_pairs << <grid, block >> > (d_input, d_temp, size);

	gpuErrchk(cudaDeviceSynchronize());

	cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost);

	int gpu_result = 0;
	for (int i = 0; i < grid.x; i++)
	{
		gpu_result += h_ref[i];
	}

	//validity check
	compare_results(gpu_result, cpu_result);

	gpuErrchk(cudaFree(d_temp));
	gpuErrchk(cudaFree(d_input));

	free(h_ref);
	free(h_input);


	gpuErrchk(cudaDeviceReset());
	return 0;
}
