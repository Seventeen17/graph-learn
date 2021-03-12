# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import random
import time
import numpy as np
import tensorflow as tf
import graphlearn as gl
from graphlearn.python.nn.tf.app.link_predictor import UnsupervisedLinkPredictor
from graphlearn.python.nn.tf.app.node_classifier import NodeClassifier
from graphlearn.python.nn.tf.data.data_flow import DataFlow
from graphlearn.python.nn.tf.layers.ego_sage_layer import EgoSAGELayer
from graphlearn.python.nn.tf.layers.ego_sage_layer import EgoSAGELayerGroup
from graphlearn.python.nn.tf.model.ego_sage import EgoGraphSAGE, HomoEgoGraphSAGE
from graphlearn.python.nn.tf.trainer import Trainer
import graphlearn.python.tests.utils as utils

class EgoSAGETestCase(unittest.TestCase):
  """ Base class of sampling test.
  """
  @classmethod
  def setUpClass(cls):
    pass

  @classmethod
  def tearDownClass(cls):
    pass

  def setUp(self):
    pass

  def tearDown(self):
    pass

  def gen_node_labeled(self, node_type):
    def write_meta(f):
      meta = 'id:int64\tlabel:int64\tfeature:string\n'
      f.write(meta)

    def write_data(f):
      for i in range(120):
        line = '%d\t%d\t%f:%f:%f:%f\n' % (i, i % 2, i * 0.1, i * 0.2, i * 0.3, i * 0.4)
        f.write(line)

    path = '%s/%s_%d' % ('.data_path/', node_type, int(time.time() * 1000))
    with open(path, 'w') as f:
      write_meta(f)
      write_data(f)
    return path

  def test_homo_sage_supervised(self):
    item_path = self.gen_node_labeled('item')
    i2i_path = utils.gen_edge_data('item', 'item', (0, 100), (0, 100), schema=[])

    g = gl.Graph() \
          .node(item_path, 'i',
                decoder=gl.Decoder(attr_types=['float'] * 4, attr_dims=[10] * 4,
                                   labeled=True)) \
          .edge(i2i_path, ('i', 'i', 'i-i'), decoder=gl.Decoder()) \
          .init()

    query = g.V('i').batch(10).alias('i') \
             .outV('i-i').sample(5).by('topk').alias('hop1') \
             .outV('i-i').sample(5).by('random').alias('hop2') \
             .values()
    df = DataFlow(query)

    dims = np.array([4, 16, 8])
    model = HomoEgoGraphSAGE(dims, bn_fn=None, active_fn=tf.nn.relu, droput=0.1)

    eg = df.get_ego_graph('i')
    embeddings = model.forward(eg)
    nc = NodeClassifier(dims=[8, 4], class_num=2)
    logits, loss = nc.forward(embeddings, eg.nodes.labels)

    trainer = Trainer()
    trainer.minimize(loss)
    def trace(ret):
      self.assertEqual(len(ret), 2)
      self.assertEqual(list(ret[0].shape), [10, 2])
    trainer.step_to_epochs(10, [logits, loss], trace)
    trainer.close()
    g.close()


if __name__ == "__main__":
  unittest.main()
