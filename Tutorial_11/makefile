TARGET = main.exe
OBJS = main.o gpu.o

all: $(TARGET)

$(TARGET): $(OBJS)
	nvcc -O3 $(OBJS) -code=sm_80 -arch=compute_80 -o $(TARGET)

gpu.o: gpu.cu gpu.h
	nvcc -O3 -c gpu.cu -arch=compute_80 -code=sm_80 -o gpu.o

main.o: main.c gpu.h
	g++ -O3 -I/usr/local/cuda/include -c main.c -o main.o

clean:
	rm -f *.o $(TARGET)