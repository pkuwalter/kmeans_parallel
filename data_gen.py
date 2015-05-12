#!/usr/bin/env python

import sys
import random
import cStringIO
import struct

def gen_data(num_points,dimension,output_file):
    buffer = cStringIO.StringIO()
    for i in range(num_points):
        buffer.write(str(i))
        buffer.write(' ')
        for j in range(dimension):
            buffer.write(str(random.random()))
            buffer.write(' ')
        buffer.write("\n")

    with open(output_file, "w+") as f:
        f.write(buffer.getvalue())
    buffer.close()

def usage():
    printf("Usage: %s [num_points] [dimention] [output_file]" % (sys.argv[0]))
    sys.exit(1)
    
if (len(sys.argv) != 4):
    usage();
num_points = int(sys.argv[1])
dimension = int(sys.argv[2])
output_file = sys.argv[3]
gen_data(num_points, dimension, output_file)
