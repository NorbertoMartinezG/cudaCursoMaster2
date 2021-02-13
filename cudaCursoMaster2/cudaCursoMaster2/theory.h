/* ----------------------------- 232 CUDA Dynamic parallelism -----------------------------------
	- Lanza kernel desde la misma GPU

	----------------------------- 233 Reduction with dynamic parallelism -----------------------------------
	- Lanza kernel recursivo en GPU.

	----------------------------- 234 Summary -----------------------------------

	- Warp execution
	- Resource partition and latency hiding
	- Optimizing a CUDA program based on CUDA execution model.
	- warp is the basic unit of execution in a CUDA program
	- warp divergence will hinder the performance of a CUDA program
	- synchronization between threads with in thread block using ( __syncthread() ) 

	- parallel reduction algorithm
		- Naive neighbored pairs approach / improved version
		- Interleaved pair approach
		- Data block unrolling
		- Warp unrolling
		- completely unrolling
		- Template functions

	- dynamic parallelismo
		- We can launch CUDA kernel form another kernel



	----------------------------- 235 Memory model -----------------------------------
	- gld_efficiency					- eficiencia de la carga global de la memoria
	- gld_throughput					- eficiencia de la carga global de la memoria desde el kernel
	- gld_transaction					- indica cuantas transacciones de carga de memoria global realiza el kernel
	- gld_transaction_per_request		- inica cuantas transacciones de memoria se necesitan para atneder la solicitud de memoria

	-locality
	Memorias segun su velocidad y tamaño
	Registers       fast-small
	caches			fast- small
	main memory		slow-big
	disk memory		slow-big


	----------------------------- 236 Memory types -----------------------------------

	- Registros				-son los recursos mas rapidos y escasos en GPU	
	- Registros	spills		- if a kernel uses more registers that the hardware limit, 
	- Local memory			- 
	- shared memory			- momoria en chip
	- constant memory
	- texture memory
	- global memory
	- gpu caches

	----------------------------- 237 Memory management and pinned memory -----------------------------------

	 - Host
		- Malloc
		- free

	- Device
		- cudaMalloc
		- cudaFree
		- cudaMemCpy

	- Pinned memory

	memoria paginada -> pinned memory ->DRAM(GPU)

	cudaError_t cudaMallocHost(void ** devPtr, size_t count);
	cudaFreeHost(void * ptr);

	---------------------------------------- 238 Zero copy memory --------------------------------------------

	- cudaHostAllocDefault

	- cudaHostAllocPortable

	- cudaHostAllocWriteCombined

	- cudaHostAllocMapped 

	- cudaError_t cudaHostGetDevicePointer

	Si necesita mas memoria que la memoria disponible en su GPU, la memoria de cpia cero sera una buena 
	opcion

	---------------------------------------- 239 Unified Memory --------------------------------------------

	__device__ __managed__ int y;
	cudaMallocManaged(vodi** devPtr size *size, unsigned int flags = 0);

	- se tiene un costo adicional cuando se usa memoria unificada


*/