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
from IngeoDash.config import CONFIG, Config
from IngeoDash.app import user, label_column
from EvoMSA.utils import MODEL_LANG
from dash import dcc
import numpy as np
import dash_bootstrap_components as dbc
import base64
import io
import json


def read_json(mem: Config, data):
    _ = io.StringIO(data.decode('utf-8'))
    return [json.loads(x) for x in _]


def upload(mem: Config, content, lang='es', 
           type='json', text='text', label='klass',
           call_next=label_column):
    def _label(x):
        if mem.label_header in x:
            ele = x[mem.label_header]
            if ele is not None and len(f'{ele}'):
                return True
        return False
    mem.mem.update(dict(label_header=label, text=text))
    mem.label_header = label
    mem.text = text
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    data = globals()[f'read_{type}'](mem, decoded)
    username, db = user(mem)
    labels = np.unique([x[mem.label_header]
                        for x in data if _label(x)])
    permanent = db.get(mem.permanent, list())    
    if labels.shape[0] > 1:
        original = [x for x in data if not _label(x)]
        permanent.extend([x for x in data if _label(x)])
    else:
        original = data
    db[mem.data] = original[:mem.n_value]
    db[mem.permanent] = permanent
    db[mem.original] = original[mem.n_value:]
    mem.mem.update({mem.lang: lang,
                    mem.size: len(data),
                    mem.username: username})
    if call_next is not None:
        call_next(mem)
    return json.dumps(mem.mem)


def upload_component():
    lang = dbc.Select(id=CONFIG.lang, value='es',
                      options=[dict(label=x, value=x) for x in MODEL_LANG])
    upload = dbc.InputGroup([dbc.InputGroupText('Language:'),
                             lang, 
                             dbc.InputGroupText('Text Column:'),
                             dcc.Input(id=CONFIG.text,
                                       value='text',
                                       type='text'),
                             dbc.InputGroupText('Text Label:'),
                             dcc.Input(id=CONFIG.label_header,
                                       value='klass',
                                       type='text'),
                             dcc.Upload(id=CONFIG.upload, 
                                        children=dbc.Button('Upload'))])
    return upload

