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
from dataclasses import dataclass, field
from sklearn.svm import LinearSVC
from typing import Tuple
from copy import deepcopy
import json


@dataclass
class Config:
    store: str = 'store'
    labels: str = 'labels'
    label_header: str = 'klass'
    data: str = 'corpus'
    permanent: str = 'permanent'
    original: str = 'original'
    next: str = 'next'
    next_label: str = 'Next'
    n_value: int = 10
    size: str = 'size'
    progress: str = 'progress'
    download: str = 'download'
    filename: str = 'filename'
    save: str = 'save'
    center: str = 'center'
    upload: str = 'upload'
    lang: str = 'lang'
    n_jobs: int = 1
    denseBoW: dict = field(default_factory=dict)
    db: dict = field(default_factory=dict)
    username: str = 'username'
    text: str = 'text'
    mem: dict = field(default_factory=dict)
    prev: str='previous'
    batch_size: str='n_value'
    checklist: str='checklist'
    active_learning: str='active_learning'
    shuffle: str='shuffle'
    labels_proportion: str='labels_proportion'
    voc_size_exponent: int=15
    voc_selection: str='most_common_by_type'
    estimator_class: object=LinearSVC
    decision_function_name: str='decision_function'
    dense_select: bool=True


    def __getitem__(self, key):
        return self.mem[key]
    
    def __setitem__(self, key, value):
        self.mem[key] = value

    def __call__(self, value):
        cls = deepcopy(self)
        if value is not None:
            cls.mem = json.loads(value) if isinstance(value, str) else value
        for key in ['label_header', 'text', 'n_value',
                    'voc_size_exponent', 'voc_selection',
                    'estimator_class', 'decision_function_name',
                    'dense_select']:
            if key in cls.mem:
                setattr(cls, key, cls.mem[key])
        return cls
    
    def __contains__(self, key):
        return key in self.mem
    
    def pop(self, *args, **kwargs):
        return self.mem.pop(*args, **kwargs)
    
    def get(self, *args, **kwargs):
        return self.mem.get(*args, **kwargs)
    

CONFIG = Config()