#!/usr/bin/env python2

import os
import re
import sys
import numpy as np

regex = re.compile("(.+)_(\d+)x\d+_(\d+)_(\d+).bin")
array = None
name = ""
resolution = 0

for filename in sys.argv[1:]:
    m = regex.match(filename)
    if m is None:
        print filename, "fail"
        continue
        
    name = m.groups()[0]
    resolution = int(m.groups()[1])
    x = int(m.groups()[2])
    y = int(m.groups()[3])
    
    if array is None:
        array = np.zeros((resolution, resolution), dtype=np.float64)
        
    part = np.fromfile(filename, dtype=np.float64)
    part_size = np.sqrt(part.shape[0])
    part.resize((part_size, part_size))
    
    array[x:x+part_size, y:y+part_size] = part
    print filename

array[resolution/2:resolution, :] = np.float64(1.0) - array[resolution/2-1::-1, :]
array.tofile("%s_%ix%i.bin" % (name, resolution, resolution))