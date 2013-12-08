DEBUG=DEBUG
CC=g++
CFLAGS=-Wall -O0 -g3 -D$(DEBUG)
INCLUDE=-I/usr/local/cuda-5.0/include
LIBS=-L/usr/local/cuda-5.0/lib64 -lcuda -lcudart

NVCC=nvcc
NVCCFLAGS=-c -arch=sm_20 -D$(DEBUG)

OBJS=file_io.o kmeans.o main.o

.PHONY: default kmeans-gpu clean

default: clean kmeans-gpu

kmeans-gpu: $(OBJS)
	$(CC) $(CFLAGS) $(LIBS) $(OBJS) -o kmeans-gpu

main.o: main.cpp
	$(CC) -c $(CFLAGS) $< -o $@

file_io.o: file_io.cpp
	$(CC) -c $(CFLAGS) $< -o $@

kmeans.o: kmeans.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f *.o *~ kmeans-gpu
