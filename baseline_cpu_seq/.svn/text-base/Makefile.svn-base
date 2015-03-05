DEBUG=DEBUG
TIMING=TIMING
CC=g++
CFLAGS=-Wall -O0 -g3 -D$(DEBUG) -D$(TIMING)
LIBS=

OBJS=file_io.o kmeans.o mtime.o main.o

.PHONY: default kmeans-cpu clean

default: clean kmeans-cpu

kmeans-cpu: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o kmeans-cpu

main.o: main.cpp
	$(CC) -c $(CFLAGS) $< -o $@

file_io.o: file_io.cpp
	$(CC) -c $(CFLAGS) $< -o $@

mtime.o: mtime.cpp
	$(CC) -c $(CFLAGS) $< -o $@

kmeans.o: kmeans.cpp
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f *.o *.d *~ kmeans-cpu
