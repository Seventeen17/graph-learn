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

  def gen_user(self):
    def write_meta(f):
      meta = 'id:int64\tfeature:string\n'
      f.write(meta)

    def write_data(f):
      for i in range(120):
        line = '%d\t%f:%f:%f:%f\n' % (i, i * 0.1, i * 0.2, i * 0.3, i * 0.4)
        f.write(line)

    path = '%s/%s_%d' % ('.data_path/', 'user', int(time.time() * 1000))
    with open(path, 'w') as f:
      write_meta(f)
      write_data(f)
    return path
  
  def gen_item(self):
    def write_meta(f):
      meta = 'id:int64\tfeature:string\n'
      f.write(meta)

    def write_data(f):
      for i in range(120):
        line = '%d\t%f:%f:%d:%s\n' % (i, i * 0.1, i * 0.2, i, str(i))
        f.write(line)

    path = '%s/%s_%d' % ('.data_path/', 'item', int(time.time() * 1000))
    with open(path, 'w') as f:
      write_meta(f)
      write_data(f)
    return path

  def test_heter_sage_unsupervised(self):
    user_path = self.gen_user()
    item_path = self.gen_item()
    u2i_path = utils.gen_edge_data('user', 'item', (0, 100), (0, 100), schema=[])
    i2i_path = utils.gen_edge_data('item', 'item', (0, 100), (0, 100), schema=[])

    user_attr_types = ['float'] * 4
    item_attr_types = ['float', 'float', ('string', 10), ('string', 10)]
    attr_dims=[10] * 4

    g = gl.Graph() \
          .node(user_path, 'u', decoder=gl.Decoder(attr_types=user_attr_types, attr_dims=attr_dims)) \
          .node(item_path, 'i', decoder=gl.Decoder(attr_types=item_attr_types, attr_dims=attr_dims)) \
          .edge(u2i_path, ('u', 'i', 'u-i'), decoder=gl.Decoder()) \
          .edge(i2i_path, ('i', 'i', 'i-i'), decoder=gl.Decoder()) \
          .init()

    query = g.E('u-i').batch(10).alias('seed').each(lambda e: (
      e.inV().alias('i').outV('i-i').sample(15).by('topk').alias('dst_hop1').outV('i-i').sample(10).by('topk').alias('dst_hop2'),
      e.outV().alias('u').each(lambda v: (
        v.outV('u-i').sample(15).by('edge_weight').alias('src_hop1').outV('i-i').sample(10).by('topk').alias('src_hop2'),
        v.outNeg('u-i').sample(5).by('in_degree').alias('neg').outV('i-i').sample(15).by('topk').alias('neg_hop1')\
          .outV('i-i').sample(10).by('topk').alias('neg_hop2'))))) \
      .values()
    df = DataFlow(query)

    src_dim = 4
    dst_dim = 22
    layer_ui = EgoSAGELayer("heter_ui",
                            input_dim=(src_dim, dst_dim),
                            output_dim=12,
                            agg_type="mean",
                            com_type="concat")
    layer_ii = EgoSAGELayer("heter_ii",
                            input_dim=dst_dim,
                            output_dim=12,
                            agg_type="mean",
                            com_type="concat")
    layer_uii = EgoSAGELayer("heter_uii",
                             input_dim=(12, 12),
                             output_dim=8,
                             agg_type="sum",
                             com_type="concat")
    layer_iii = EgoSAGELayer("heter_iii",
                             input_dim=(12, 12),
                             output_dim=8,
                             agg_type="sum",
                             com_type="concat")
    layer_group_1 = EgoSAGELayerGroup([layer_ui, layer_ii])
    layer_group_2 = EgoSAGELayerGroup([layer_uii])
    src_model = EgoGraphSAGE(
        [layer_group_1, layer_group_2],
        bn_fn=None,
        active_fn=tf.nn.relu,
        droput=0.1)

    layer_group_3 = EgoSAGELayerGroup([layer_ii, layer_ii])
    layer_group_4 = EgoSAGELayerGroup([layer_iii])
    dst_model = EgoGraphSAGE(
        [layer_group_3, layer_group_4],
        bn_fn=None,
        active_fn=tf.nn.relu,
        droput=0.1)

    src_embeddings = src_model.forward(df.get_ego_graph('u'))
    dst_embeddings = dst_model.forward(df.get_ego_graph('i'))
    neg_embeddings = dst_model.forward(df.get_ego_graph('neg'))
    neg_embeddings = tf.reshape(neg_embeddings, [-1, 5, 8])

    lp = UnsupervisedLinkPredictor(name="unlp", dims=[8, 4])
    loss = lp.forward(src_embeddings, dst_embeddings, neg_embeddings)

    trainer = Trainer()
    trainer.minimize(loss)
    def trace(ret):
      self.assertEqual(len(ret), 3)
      self.assertEqual(list(ret[0].shape), [10, 8])
      self.assertEqual(list(ret[1].shape), [10, 8])
    trainer.step_to_epochs(10, [src_embeddings, dst_embeddings, loss], trace)
    trainer.close()
    g.close()


if __name__ == "__main__":
  unittest.main()
