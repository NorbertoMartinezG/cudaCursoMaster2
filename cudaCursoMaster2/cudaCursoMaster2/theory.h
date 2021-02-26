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


	---------------------------------------- 240 memory access patterns --------------------------------------
		-L1
	sm       global memory
		-L2

	1 thread = 4 bytes
	1 warp = 32 threads
	128 bytes transaction 


	- Aligned memory accesses
		- first address is an even multiple of the cache granularity
	- Coalesced memory accesses
		- 32 threads in a warp access a continuous chunk of memory


    ------------------------------------------- 241 Global memory writes --------------------------------------

	- La escritura alineada de memoria aumenta el rendimiento del programa

	------------------------------------------- 242 AOS vs SOA ------------------------------------------------

	AOS - Matriz de estructuras 50% eficiencia

		struct testStruct
		{
			float x;
			float y;
		}

		struct testStruct AoS[N]


	SOA - Estructura de patron de matriz 100% eficiencia

		struct testStruct
		{
			float x[N];
			float y[N];
		}

		struct testStruct SOA;

	------------------------------------------- 243 Matrix transpose ----------------------------------------

	- transposicion de matriz -- cambio de columnas por filas 

	0	4	8				0 1  2  3
	1	5	9       =>		4 5  6  7
	2	6	10				8 9 10 11
	3	7	11

	0 4 8 1 5 9 2 6 10 3 7 11

	se agregan punteros
	0 1 2 3 4 5 6 7  8  9 10  11
	0 4 8 1 5 9 2 6 10  3  7  11

	se utilizan indices basados en la matriz para conocer su ubicacion antes y despues de la transposicion
	-nx= 4, ny= 3 = matriz de 4 columnas por 3 filas
	-ix= 2, iy= 1 = posicion del elemento que se desea conocer su ubicacion antes y despues de la transposicion 

	out = ix*ny+iy Posicion despues de la transposicion
	in  = iy*nx+ix Posicion antes de la transposicion

	---------------------------------------- 244 Matrix transpose with unrolling ---------------------------


	-------------------------  245 Matrix transpose with diagonal coordinate system ------------------------

	- mejora el rendimiento

	-------------------------  246 summary ------------------------
	- Tipos de memoria
	- Ubicacion y latencias
	- Declaracion de tipos de memoria compartida y registros
	- Pinned memory
	- Unified memory de hosty y el device
	- Global memory access pattern
		- Align memory access
		- Coalesced memory access
	- AOS vs SOA
	- Matriz transpuesta


	------------------------- 447 Introduction to CUDA shared memory ----------------------------------------

	- compute capability	2.0		2.1		3.0		3.2		3.5		3.7		5.0		5.2		5.3		6.0		6.1		6.2
	- shared memory(bytes)	48		48		48		48		48		112		64		96		64		96		64

	- memoria compartida
	- memoria global

	------------------------- 448 Shared memory access modes and memory banks -------------------------------
	- shared memory bank
	- bank width is 64 bits and access mode is 32 bit

		B1		B2		B3...		B31 -> bancos de memoria 32 bit
		0,32	1,33	2,34...		31,63

		B1		B2		B3...		B31 -> bancos de memoria 32 bit
		0,2		3,4		5,6...		62,63

	------------------------- 449 Shared memory access modes and memory banks -------------------------------
	- acceso a la memoria en columna - requiere multiples transacciones
	- acceso a la memoria en fila - requiere una solicitud de transaccion

*/