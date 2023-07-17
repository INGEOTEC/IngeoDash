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
from EvoMSA import BoW, DenseBoW
from typing import Union
from IngeoDash.config import Config
from IngeoDash.config import CONFIG
import numpy as np


def model(mem: Config, data: dict):
    lang = mem[mem.lang]
    if lang not in CONFIG.denseBoW:
        dense = DenseBoW(lang=lang, voc_size_exponent=15,
                         n_jobs=mem.n_jobs, dataset=False)
        CONFIG.denseBoW[lang] = dense.text_representations
    dense = DenseBoW(lang=lang, key=mem.text,
                     label_key=mem.label_header,
                     voc_size_exponent=15,
                     n_jobs=mem.n_jobs,
                     dataset=False, emoji=False, keyword=False)
    dense.text_representations_extend(CONFIG.denseBoW[lang])
    return dense.select(D=data).fit(data)


def label_column_predict(mem: Config, model=None):
    db = CONFIG.db[mem[mem.username]]
    data = db[mem.data]
    if len(data) == 0:
        return   
    D = db[mem.permanent]
    dense = model(mem, D)
    hys = dense.predict(data).tolist()
    for ele, hy in zip(data, hys):
        ele[mem.label_header] = ele.get(mem.label_header, hy)        


def label_column(mem: Config, model=model):
    db = CONFIG.db[mem[mem.username]]
    if mem.permanent in db:
        _ = np.unique([x[mem.label_header]
                       for x in db[mem.permanent]])
        if _.shape[0] > 1:
            mem[mem.labels] = tuple(_.tolist())
            return label_column_predict(mem, model=model)
    label = mem.get(mem.labels, ('-', ))[0]
    data = db[mem.data]
    for ele in data:
        ele[mem.label_header] = ele.get(mem.label_header, label)


def flip_label(mem: Config, k: int):
    db = CONFIG.db[mem[mem.username]]
    data = db[mem.data]
    assert k < len(data)
    labels = mem.get(mem.labels, ('-', '+')) 
    label = data[k][mem.label_header]
    index = (labels.index(label) + 1) % len(labels)
    data[k][mem.label_header] = labels[index]
    return data[k]


def store(mem: Config):
    db = CONFIG.db[mem[mem.username]]
    data = db.pop(mem.data) if mem.data in db else []
    try:
        permanent = db[mem.permanent]
    except KeyError:
        permanent = []
    permanent.extend(data)        
    db[mem.permanent] = permanent


def similarity(query: Union[list, str],
               dataset: list, key: str='text',
               lang: str='es'):
    if isinstance(query, str):
        query = [query]
    trans = BoW(lang=lang, key=key).transform
    query = trans(query)
    dataset = trans(dataset)
    return dataset.dot(query.T).toarray()