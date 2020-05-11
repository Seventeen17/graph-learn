from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

def gen_files(path, u_count, i_count):
    with open(os.path.join(path, "node_table"), 'w') as f:
        s = 'id:int64\tlabel:int64\tfeature:string\n'
        f.write(s)
        for i in range(0, u_count):
            line = '%d\t%d\t' % (i, 1 % 7)
            for j in range(224):
                line += '%f:' % (i * 0.1)
            line += '0.1\n'
            f.write(line)

    with open(os.path.join(path, "edge_table"), 'w') as f:
        s = 'src_id:int64\tdst_id:int64\tweight:float\n'
        f.write(s)
        for i in range(u_count):
            for j in range(0, i_count):
                s = '%d\t%d\t%f\n' % (i, j, (i + j) * 0.1)
                f.write(s)

    with open(os.path.join(path, "train_table"), 'w') as f:
        s = 'id:int64\tweight:float\n'
        f.write(s)
        for i in range(0, 140):
            line = '%d\t%f\n' % (i, 1.0)
            f.write(line)

    with open(os.path.join(path, "test_table"), 'w') as f:
        s = 'id:int64\tweight:float\n'
        f.write(s)
        for i in range(0, 1000):
            line = '%d\t%f\n' % (i, 1.0)
            f.write(line)

    with open(os.path.join(path, "val_table"), 'w') as f:
        s = 'id:int64\tweight:float\n'
        f.write(s)
        for i in range(0, 300):
            line = '%d\t%f\n' % (i, 1.0)
            f.write(line)

gen_files("data", 10000, 100) 
 32  examples/tf/aggregator/gen_test_data.py 
@@ -0,0 +1,32 @@
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

def gen_files(path, u_count, i_count):
    with open(os.path.join(path, "data_{}_{}/user".format(u_count, i_count)), 'w') as f:
        s = 'id:int64\tweight:float\n'
        f.write(s)
        for i in range(u_count):
            s = '%d\t%f\n' % (i, i / 10.0)
            f.write(s)

    with open(os.path.join(path, "data_{}_{}/item".format(u_count, i_count)), 'w') as f:
        s = 'id:int64\tfeature:string\n'
        f.write(s)
        for i in range(u_count, u_count + i_count):
            line = '%d\t' % (i)
            for j in range(224):
                line += '%f:' % (i * 0.1)
            line += '0.1\n'
            f.write(line)

    with open(os.path.join(path, "data_{}_{}/u-i".format(u_count, i_count)), 'w') as f:
        s = 'src_id:int64\tdst_id:int64\tweight:float\n'
        f.write(s)
        for i in range(u_count):
            for j in range(u_count, u_count + i_count):
                s = '%d\t%d\t%f\n' % (i, j, (i + j) * 0.1)
                f.write(s) 