# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Distributed training script for supervised GraphSage.
This simple example uses two machines and each has one TensorFlow worker and ps.
Graph-learn client is colocate with TF worker, and server with ps.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np

import graphlearn as gl
import tensorflow as tf

from gen_test_data import gen_files

# tf settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("task_index", None, "Task index")
flags.DEFINE_string("job_name", None, "worker or ps")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("tracker", '/mnt/data/nfs/graph-learn/aggregator/','tracker dir')

# Note: tracker dir should be cleaned up before training.
# graphlearn settings
client_count = len(FLAGS.worker_hosts.split(","))
server_count = len(FLAGS.ps_hosts.split(","))

graph_cluster = {"client_count": client_count, "tracker": FLAGS.tracker,
                 "server_count": server_count}


def load_graph(src_count, dst_count):
  data_path = os.path.join(FLAGS.tracker, "data_{}_{}".format(src_count, dst_count))
  if not os.path.exists(data_path):
    os.system('mkdir -p {}'.format(data_path))
    gen_files(FLAGS.tracker, src_count, dst_count)

  g = gl.Graph()

  g.node(os.path.join(FLAGS.tracker, "data_{}_{}/user".format(src_count, dst_count)),
         node_type="user", decoder=gl.Decoder(weighted=True)) \
    .node(os.path.join(FLAGS.tracker, "data_{}_{}/item".format(src_count, dst_count)),
          node_type="item", decoder=gl.Decoder(attr_types=['float'] * 225)) \
    .edge(os.path.join(FLAGS.tracker, "data_{}_{}/u-i".format(src_count, dst_count)),
          edge_type=("user", "item", "buy"), decoder=gl.Decoder(weighted=True))
  return g

def train(graph, dst_count, pre_agg=False):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  server = tf.train.Server(cluster, FLAGS.job_name, task_index=FLAGS.task_index)

  batch_size = 32
  num_float_attrs = 225
  n_epoches = 10
  if FLAGS.job_name == 'worker': # also graph-learn client in this example.

    s1 = graph.node_sampler("user", batch_size=batch_size)
    s2 = graph.neighbor_sampler("buy", expand_factor=1, strategy="full")

    with tf.device(tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d" % (FLAGS.task_index), cluster=cluster)):

      num_nbrs = server_count if pre_agg else dst_count
      neigh_vecs = tf.placeholder(tf.float32, [None, num_nbrs, num_float_attrs])
      vecs = tf.reduce_mean(neigh_vecs, axis=1)

      with tf.train.MonitoredTrainingSession(master=server.target,
                                             is_chief=(FLAGS.task_index == 0)) as mon_sess:

        for i in range(n_epoches):
          start_time = time.time()
          while not mon_sess.should_stop():
            try:
              seed_nodes = s1.get().ids
              print(seed_nodes[-1])
              nbr_nodes = s2.get(seed_nodes).layer_nodes(1)
              feed = nbr_nodes.embedding_agg(func="mean") if pre_agg else np.reshape(nbr_nodes.float_attrs, (-1, dst_count, num_float_attrs))
              mon_sess.run(vecs, feed_dict={neigh_vecs: feed})
            except gl.OutOfRangeError:
              break
          print("Epoch {}, without agg, time : {}s".format(i, time.time()-start_time))
  else:
    server.join()

def main():
  src_count = 100000
  dst_count = 20
  pre_agg = True
  print("main")
  g = load_graph(src_count, dst_count)
  g_role = "server"
  if FLAGS.job_name == "worker":
    g_role = "client"
  g.init(cluster=graph_cluster, job_name=g_role, task_index=FLAGS.task_index)
  train(g, dst_count, pre_agg=pre_agg)

if __name__ == "__main__":
  main()