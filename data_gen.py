#!/usr/bin/env python

import sys
import random
import cStringIO
import struct

def main():
    num_points = 640
    dimension = 5

    buffer = cStringIO.StringIO()
    for i in range(num_points):
        buffer.write(str(i))
        buffer.write(' ')
        for j in range(dimension):
            buffer.write(str(random.random()))
            buffer.write(' ')
        buffer.write("\n")

    with open("input.dat", "w") as f:
        f.write(buffer.getvalue())
    buffer.close()

                             
    

if "__main__" == __name__:
    main()
