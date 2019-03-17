# Optimizing-Parallel-Reduction
Optimizing summing up elements in huge vector using CUDA techniques. To do this I use one of algortihms proposed by NVIDIA Corporation. 
Elements are being added up in shared memory in parts. Threads in CUDA share their memory in share memory, which can be optimized in very efficient way. 
For comparsion float and doubles elements of vector are added up on CPU and GPU architecture and time is measured.
To sum up I have very nice results comparing CPU to GPU to manage the addition. For very huge vectors(where all my memory is occupied - 8GB RAM) like 100 bilions of doubles and floats: GPU seems to be 3k - 4k faster.

To run the project you need to have CUDA installed and use Visual Studio 2015(v140) compiler. 
