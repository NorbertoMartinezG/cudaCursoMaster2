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


*/