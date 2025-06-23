all: GPU CPU
	nvcc -O3 main.o gpu.o -code=sm_80 -arch=compute_80 -o main.exe
GPU:
	nvcc -O3 gpu.cu -c -arch=compute_80 -code=sm_80
CPU:
	g++ -O3 main.c -c
clean:
	rm *.o main.exe