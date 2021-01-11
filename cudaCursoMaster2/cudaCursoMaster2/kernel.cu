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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void print_details_of_warps()
{
	int gid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

	int warp_id = threadIdx.x / 32;

	int gbid = blockIdx.y * gridDim.x + blockIdx.x;

	printf( "tid: %d, bid.x: %d, bid.y: %d, gid: %d, warp_id: %d, gbid: %d \n",
		threadIdx.x, blockIdx.x, blockIdx.y, gid, warp_id, gbid);
}


int main(int argc, char** argv) {

	dim3 block_size(42);
	dim3 grid_size(2, 2);

	print_details_of_warps << <grid_size, block_size >> > ();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return EXIT_SUCCESS;

}


