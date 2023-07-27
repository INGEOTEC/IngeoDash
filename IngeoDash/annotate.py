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
from EvoMSA import BoW, DenseBoW, StackGeneralization
from typing import Union
from IngeoDash.config import Config
from IngeoDash.config import CONFIG
from sklearn.svm import LinearSVC
import numpy as np


def has_label(mem: Config, x):
    if mem.label_header in x:
        ele = x[mem.label_header]
        if ele is not None and len(f'{ele}'):
            return True
    return False


def model_bow(mem: Config, data: dict):
    lang = mem[mem.lang]
    return BoW(lang=lang, key=mem.text,
               label_key=mem.label_header,
               voc_size_exponent=mem.voc_size_exponent,
               voc_selection=mem.voc_selection).fit(data)


def model(mem: Config, data: dict):
    lang = mem[mem.lang]
    if lang not in CONFIG.denseBoW:
        dense = DenseBoW(lang=lang, voc_size_exponent=mem.voc_size_exponent,
                         voc_selection=mem.voc_selection,
                         n_jobs=mem.n_jobs, dataset=False)
        CONFIG.denseBoW[lang] = dense.text_representations
    dense = DenseBoW(lang=lang, key=mem.text,
                     label_key=mem.label_header,
                     voc_size_exponent=mem.voc_size_exponent,
                     voc_selection=mem.voc_selection,
                     n_jobs=mem.n_jobs,
                     dataset=False, emoji=False, keyword=False)
    dense.text_representations_extend(CONFIG.denseBoW[lang])
    if mem.dense_select:
        dense.select(D=data)
    _ = np.unique([x[mem.label_header] for x in data],
                  return_counts=True)[1]
    if np.any(_ < 5):
        return dense.fit(data)
    bow = BoW(lang=lang, key=mem.text,
              label_key=mem.label_header,
              voc_size_exponent=mem.voc_size_exponent,
              voc_selection=mem.voc_selection)
    stack = StackGeneralization(decision_function_models=[bow, dense],
                                decision_function_name=mem.decision_function_name,
                                estimator_class=mem.estimator_class)
    return stack.fit(data)


def _select_top_k(selected, members, cnt):
    if cnt <= 0:
        return
    sel = []
    for i in members:
        if i not in selected:
            sel.append(i)
        if len(sel) == cnt:
            break
    [selected.add(x) for x in sel]
    

def balance_selection(mem: Config, hy):
    db = CONFIG.db[mem[mem.username]]
    D = db[mem.permanent]
    klasses, n_elements = np.unique([x[mem.label_header] for x in D],
                                    return_counts=True)
    missing = 1 - n_elements / n_elements.max()
    max_ele = n_elements.max()
    selected = set()
    if klasses.shape[0] > 2:
        ss = np.argsort(hy, axis=0)[::-1].T
        for k, (c, n_ele, s) in enumerate(zip(mem.n_value * missing,
                                              max_ele - n_elements, ss)):
            cnt = np.ceil(min(c, n_ele)).astype(int)
            _select_top_k(selected, s, cnt)
    else:
        ss = np.argsort(hy, axis=0)[:, 0]
        ss = [ss, ss[::-1]]
        for c, s in zip(max_ele - n_elements, ss):
            _select_top_k(selected, s, c)
    cnt = np.ceil((mem.n_value - len(selected)) / klasses.shape[0]).astype(int)
    [_select_top_k(selected, s, cnt) for s in ss]
    selected = list(selected)
    np.random.shuffle(selected)
    selected = np.array(sorted(selected[:mem.n_value]))
    if klasses.shape[0] > 2:
        _k = klasses[hy[selected].argmax(axis=1)]
    else:
        _k = klasses[np.where(hy[:, 0][selected] > 0, 1, 0)]
    return selected, _k


def random_selection(mem: Config, hy):
    index = np.arange(hy.shape[0])
    np.random.shuffle(index)
    index = index[:mem.n_value]
    index.sort()
    labels = np.array(mem[mem.labels])
    klasses = labels[np.where(hy[:, 0][index] > 0, 1, 0)]
    return index, klasses    


def boundary_selection(mem: Config, hy):
    if len(mem[mem.labels]) > 2:
        index = np.arange(hy.shape[0])
        ss = np.argsort(hy, axis=1)
        diff = hy[index, ss[:, -1]] - hy[index, ss[:, -2]]
        index = np.argsort(diff)[:mem.n_value]
        index.sort()
        labels = np.array(mem[mem.labels])
        klasses = labels[hy[index].argmax(axis=1)]
    else:
        index = np.argsort(np.fabs(hy[:, 0]))[:mem.n_value]
        index.sort()
        labels = np.array(mem[mem.labels])
        klasses = labels[np.where(hy[:, 0][index] > 0, 1, 0)]
    return index, klasses    


def active_learning_selection(mem: Config):
    db = CONFIG.db[mem[mem.username]]
    dense = model(mem, db[mem.permanent])  
    D = db[mem.data] + db.get(mem.original, list())
    hy = dense.decision_function(D)
    index, klasses = globals()[f'{mem.active_learning_selection}'](mem, hy)
    data = []
    for cnt, i in enumerate(index):
        ele = D.pop(i - cnt)
        ele[mem.label_header] = ele.get(mem.label_header, klasses[cnt])
        data.append(ele)
    db[mem.original] = D
    db[mem.data] = data
    return dense


def label_column_predict(mem: Config):
    db = CONFIG.db[mem[mem.username]]
    data = db[mem.data]
    if len(data) == 0 or np.all([has_label(mem, x) for x in data]):
        return   
    if mem.active_learning in mem and mem[mem.active_learning]:
        return active_learning_selection(mem)
    D = db[mem.permanent]
    dense = globals()[f'{mem.model}'](mem, D)   
    hys = dense.predict(data).tolist()
    for ele, hy in zip(data, hys):
        ele[mem.label_header] = ele.get(mem.label_header, hy)
    return dense        


def label_column(mem: Config):
    db = CONFIG.db[mem[mem.username]]
    if mem.permanent in db:
        _ = np.unique([x[mem.label_header]
                       for x in db[mem.permanent]])
        if _.shape[0] > 1:
            mem[mem.labels] = tuple(_.tolist())
            return label_column_predict(mem)
    label = mem.get(mem.labels, (0, ))[0]
    data = db[mem.data]
    for ele in data:
        ele[mem.label_header] = ele.get(mem.label_header, label)


def flip_label(mem: Config, k: int):
    db = CONFIG.db[mem[mem.username]]
    data = db[mem.data]
    assert k < len(data)
    labels = mem.get(mem.labels, (0, 1)) 
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