#!/usr/bin/env python2

import sys
import numpy as np
import matplotlib.pyplot as plt

def convert_c(name):
    cbresult = np.fromfile("%s" % name, dtype=np.float64)
    cbsize = np.sqrt(cbresult.shape[0])
    cbresult = cbresult.reshape((cbsize, cbsize))
    print name
    print np.min(cbresult)
    print np.max(cbresult)
    print
    
    vmin = np.min(cbresult)
    vmax = np.max(cbresult)
    
    if ("image" in name):
        cbresult[cbresult < 0] = 0.0
        cbresult[cbresult > 1] = 1.0
        vmin = 0.0
        vmax = 1.0
    plt.imsave("%s.png" % name[:-4], np.rot90(cbresult), cmap='RdYlBu', vmin=vmin, vmax=vmax)
    
map(convert_c, sys.argv[1:])

raw_input("OK")