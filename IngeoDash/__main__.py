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
from IngeoDash.app import table_next, progress, update_row, table, table_component, table_prev, labels_proportion
from IngeoDash.download import download, download_component
from IngeoDash.upload import upload, upload_component
from IngeoDash.config import CONFIG
from dash import dcc, Output, Input, callback, Dash, State, ctx
import dash_bootstrap_components as dbc


@callback(
    Output(CONFIG.store, 'data'),
    Input(CONFIG.next, 'n_clicks'),
    Input(CONFIG.prev, 'n_clicks'),    
    State(CONFIG.store, 'data'),
    prevent_initial_call=True
)
def table_next_callback(next, prev, mem):
    mem = CONFIG(mem)
    if ctx.triggered_id == mem.next:
        return table_next(mem)
    return table_prev(mem)


@callback(
    Output(CONFIG.center, 'children'),
    Input(CONFIG.store, 'data'),
    prevent_initial_call=True    
)
def table_callback(mem):
    mem = CONFIG(mem)
    return table(mem)


@callback(
    Output(CONFIG.progress, 'value'),
    Input(CONFIG.store, 'data')
)
def progress_callback(mem):
    mem = CONFIG(mem)
    return progress(mem)


@callback(
    Output(CONFIG.labels_proportion, 'children'),
    Input(CONFIG.store, 'data')
)
def progress_callback(mem):
    mem = CONFIG(mem)
    return labels_proportion(mem)


@callback(
    Output(CONFIG.data, 'data'),
    Input(CONFIG.data, 'active_cell'),
    State(CONFIG.store, 'data'),
    prevent_initial_call=True
)
def update_row_callback(table, mem):
    mem = CONFIG(mem)
    return update_row(mem, table)


@callback(Output(CONFIG.download, 'data'),
          Input(CONFIG.save, 'n_clicks'),
          State(CONFIG.filename, 'value'),
          State(CONFIG.store, 'data'),
          prevent_initial_call=True)
def download_callback(_, filename, mem):
    mem = CONFIG(mem)
    return download(mem, filename)


@callback(
    Output(CONFIG.store, 'data', allow_duplicate=True),
    Input(CONFIG.upload, 'contents'),
    State(CONFIG.lang, 'value'),
    State(CONFIG.text, 'value'),
    State(CONFIG.label_header, 'value'),
    State(CONFIG.batch_size, 'value'),
    State(CONFIG.checklist, 'value'),
    State(CONFIG.size, 'value'),
    State(CONFIG.store, 'data'),
    prevent_initial_call=True
)
def upload_callback(content, lang, text, label, 
                    n_value, checklist, size,
                    mem):
    mem = CONFIG(mem)
    checklist = checklist if checklist is not None else []
    shuffle = CONFIG.shuffle in checklist
    active_learning = CONFIG.active_learning in checklist
    size = size if active_learning else None
    return upload(mem, content, lang=lang,
                  text=text, label=label,
                  n_value=n_value, 
                  shuffle=shuffle, size=size,
                  active_learning=active_learning)


def test_component(component):
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],
               suppress_callback_exceptions=True)
    app.layout = dbc.Container([dbc.Row(component)])
    app.run_server(debug=True)


def run(debug=True, **kwargs):
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],
               suppress_callback_exceptions=True)

    app.layout = dbc.Container([dcc.Loading(children=dcc.Store(CONFIG.store),
                                            fullscreen=True),
                                dbc.Row(table_component()),
                                dbc.Row(download_component()),
                                dbc.Row(upload_component())])
    app.run_server(debug=debug)


if __name__ == '__main__':
    run()