# Copyright 2023 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from IngeoDash.annotate import label_column, flip_label, store, similarity, model
from IngeoDash.config import CONFIG
from microtc.utils import tweet_iterator
from EvoMSA.tests.test_base import TWEETS
from EvoMSA import DenseBoW
import numpy as np


def test_label_column():
    data = [dict() for i in range(3)]
    mem = CONFIG({CONFIG.username: 'xxx'})
    CONFIG.db['xxx'] = {mem.data: data}
    db = CONFIG.db['xxx']
    label_column(mem)
    for hy in data:
        assert 0 == hy[mem.label_header]
    labels = [1, 0, 0]
    for k, ele in zip(labels,
                      data):
        ele[mem.label_header] = k
    label_column(mem)
    for y, hy in zip(labels, data):
        assert y == hy[mem.label_header]


def test_label_column_predict():
    mem = CONFIG({CONFIG.username: 'xxx'})
    mem.label_header = 'klass'
    mem[mem.lang] = 'es'
    CONFIG.denseBoW['es'] = DenseBoW(lang='es', 
                                     voc_size_exponent=15,
                                     dataset=False).text_representations
    D = list(tweet_iterator(TWEETS))
    CONFIG.db['xxx'] = {}
    db = CONFIG.db['xxx']
    db[mem.permanent] = D[:10]
    db[mem.data] = [dict(text=x['text']) for x in D[10:20]]
    label_column(mem)
    assert mem[mem.labels] == ('N', 'NONE', 'P')
    hy = np.array([x[mem.label_header] for x in db[mem.data]])
    y = np.array([x[mem.label_header] for x in D[10:20]])
    assert (y == hy).mean() >= 0.2


def test_predict_active_learning():
    _ = {CONFIG.username: 'xxx', CONFIG.label_header: 'klass',
         CONFIG.lang: 'es', CONFIG.active_learning: True}
    mem = CONFIG(_)
    CONFIG.denseBoW['es'] = DenseBoW(lang='es', 
                                     voc_size_exponent=15,
                                     dataset=False).text_representations
    D = list(tweet_iterator(TWEETS))
    for k, x in enumerate(D):
        x['id'] = k
    for x in D[10:]:
        del x['klass']
    CONFIG.db['xxx'] = {mem.permanent: D[:10], mem.data: D[10:20],
                        mem.original: D[20:]}
    label_column(mem)
    db = CONFIG.db['xxx']
    permanent = db[mem.permanent]
    data = db[mem.data]
    assert [x['id'] for x in permanent] == list(range(10))
    assert [x['id'] for x in data] != list(range(10, 20))
    D = [x for x in tweet_iterator(TWEETS) 
         if x['klass'] in ['N', 'P']]
    for k, x in enumerate(D):
        x['id'] = k
    for x in D[10:]:
        del x['klass']    
    CONFIG.db['xxx'] = {mem.permanent: D[:10], mem.data: D[10:20],
                        mem.original: D[20:]}
    label_column(mem)
    db = CONFIG.db['xxx']
    permanent = db[mem.permanent]
    data = db[mem.data]
    assert [x['id'] for x in permanent] == list(range(10))
    assert [x['id'] for x in data] != list(range(10, 20))
    

def test_flip_label():
    data = [dict() for i in range(3)]
    mem = CONFIG({CONFIG.username: 'xxx'})
    CONFIG.db['xxx'] = {mem.data: data}
    db = CONFIG.db['xxx']
    label_column(mem)
    flip_label(mem, k=1)
    assert db[mem.data][1][mem.label_header] == 1


def test_store():
    data = [dict() for i in range(3)]    
    mem = CONFIG({CONFIG.username: 'xxx'})
    CONFIG.db['xxx'] = {mem.data: data}
    db = CONFIG.db['xxx']
    label_column(mem)
    flip_label(mem, k=1)
    store(mem)
    assert mem.data not in db
    assert mem.permanent in db
    db[mem.data] = [dict(hola=1) for i in range(5)]
    store(mem)
    assert len(db[mem.permanent]) == 8
    assert db[mem.permanent][-1]['hola'] == 1


def test_similarity():
    tweets = [dict(nn=tweet['text']) 
              for tweet in tweet_iterator(TWEETS)]
    sim_values = similarity('estoy que muero de', tweets, key='nn')
    _ = sorted([[tweet['nn'], sim]for tweet, (sim, ) in zip(tweets, sim_values)],
               key=lambda x: x[1],
               reverse=True)
    assert 'Me choca ahorita' in _[0][0]


def test_stack_dense():
    from EvoMSA import BoW, DenseBoW, StackGeneralization
    mem = CONFIG({CONFIG.lang: 'es'})
    D = list(tweet_iterator(TWEETS))
    m = model(mem, D[:15])
    assert isinstance(m, DenseBoW) and not isinstance(m, StackGeneralization)
    m = model(mem, D)
    assert isinstance(m, StackGeneralization)