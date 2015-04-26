DEBUG=-DDEBUG
TIMING=-DTIMING -DKERNAL_TIMING
//METHOD=-DSYNCOUNT

CC=g++
CFLAGS=-Wall -O0 -g3 $(DEBUG) $(TIMING) $(METHOD)
INCLUDE=-I/usr/local/cuda-5.0/include
LIBS=-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas

NVCC=nvcc
NVCCFLAGS=-c -arch=sm_20 $(DEBUG) $(TIMING) $(METHOD)

OBJS=file_io.o kmeans.o mtime.o main.o

.PHONY: default kmeans-gpu clean

default: clean kmeans-gpu

kmeans-gpu: $(OBJS)
	$(CC) $(CFLAGS) $(LIBS) $(OBJS) -o kmeans-gpu

main.o: main.cpp
	$(CC) -c $(CFLAGS) $< -o $@

file_io.o: file_io.cpp
	$(CC) -c $(CFLAGS) $< -o $@

mtime.o: mtime.cpp
	$(CC) -c $(CFLAGS) $< -o $@

kmeans.o: kmeans.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f *.o *~ kmeans-gpu
