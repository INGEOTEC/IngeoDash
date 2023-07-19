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
from IngeoDash.upload import upload, upload_component, active_learning_callback
from IngeoDash.config import CONFIG
from microtc.utils import tweet_iterator
from EvoMSA.tests.test_base import TWEETS
import numpy as np
import base64
import json


def test_upload():
    mem = CONFIG({CONFIG.username: 'xxx'})
    mem.label_header = 'klass'
    D = list(tweet_iterator(TWEETS))
    CONFIG.db['xxx'] = {mem.permanent: [D[-1]]}
    D1 = [dict(text=x['text']) for x in D[50:]]
    _ = [json.dumps(x) for x in D[:50] + D1]
    content_str = str(base64.b64encode(bytes('\n'.join(_),
                                       encoding='utf-8')),
                      encoding='utf-8')
    content = f'NA,{content_str}'
    _ = upload(mem, content, 'es')
    info = json.loads(_)
    db = CONFIG.db[info[mem.username]]
    assert info[mem.size] == 50 + len(D1)
    assert len(db[mem.data]) == mem.n_value
    assert len(db[mem.permanent]) == 51
    assert len(db[mem.data]) + len(db[mem.original]) + len(db[mem.permanent]) == len(D) + 1
    for a, b in zip(db[mem.data], D[50:]):
        assert a['text'] == b['text'] and mem.label_header in a
    


def test_upload_unique():
    mem = CONFIG({CONFIG.username: 'xxx'})
    D = list(tweet_iterator(TWEETS))
    for x in D:
        x['class'] = 1
    CONFIG.db['xxx'] = {}
    _ = [json.dumps(x) for x in D[:15]]
    content_str = str(base64.b64encode(bytes('\n'.join(_),
                                       encoding='utf-8')),
                      encoding='utf-8')
    content = f'NA,{content_str}'
    _ = upload(mem, content, lang='es',
               label='class', n_value=3)
    db = CONFIG.db['xxx']
    assert len(db[mem.permanent]) == 0
    assert len(db[mem.data]) + len(db[mem.original]) == 15
    assert len(db[mem.data]) == 3


def test_upload_shuffle():
    mem = CONFIG({CONFIG.username: 'xxx'})
    D = list(tweet_iterator(TWEETS))[:15]
    for k, x in enumerate(D):
        x['klass'] = k
    CONFIG.db['xxx'] = {}
    _ = [json.dumps(x) for x in D]
    content_str = str(base64.b64encode(bytes('\n'.join(_),
                                       encoding='utf-8')),
                      encoding='utf-8')
    content = f'NA,{content_str}'
    _ = upload(mem, content, lang='es',
               shuffle=1, n_value=3)
    db = CONFIG.db['xxx']
    _ = [x['klass'] for x in db[mem.permanent]]
    assert _ != list(range(len(D)))


def test_upload_labels():
    mem = CONFIG({CONFIG.username: 'xxx'})
    CONFIG.db['xxx'] = {}
    D = list(tweet_iterator(TWEETS))
    _ = [json.dumps(x) for x in D]
    content_str = str(base64.b64encode(bytes('\n'.join(_),
                                       encoding='utf-8')),
                      encoding='utf-8')
    content = f'NA,{content_str}'
    _ = upload(mem, content, lang='es')
    db = CONFIG.db['xxx']
    assert len(db[mem.data]) == 0
    assert len(db[mem.original]) == 0
    klasses = np.unique([x['klass'] for x in D])
    mem = CONFIG(_)
    print(mem[mem.labels], klasses.tolist())
    assert mem[mem.labels] == klasses.tolist()


def test_size_active_learning():
    mem = CONFIG({CONFIG.username: 'xxx'})
    mem.label_header = 'klass'
    D = list(tweet_iterator(TWEETS))
    CONFIG.db['xxx'] = {mem.permanent: [D[-1]]}
    D1 = [dict(text=x['text']) for x in D[50:]]
    _ = [json.dumps(x) for x in D[:50] + D1]
    content_str = str(base64.b64encode(bytes('\n'.join(_),
                                       encoding='utf-8')),
                      encoding='utf-8')
    content = f'NA,{content_str}'
    _ = upload(mem, content, 'es', active_learning=True, size=30)
    info = json.loads(_)
    assert info[mem.size] == 30


def test_active_learning_callback():
    assert active_learning_callback(None)
    assert not active_learning_callback([CONFIG.active_learning])
    

def test_upload_component():
    import dash_bootstrap_components as dbc
    component = upload_component()
    assert isinstance(component, dbc.Col)