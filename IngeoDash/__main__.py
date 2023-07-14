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
from IngeoDash.app import table_next, download, progress, update_row, download_component, table, table_component
from IngeoDash.upload import upload, upload_component
from IngeoDash.config import CONFIG
from dash import dcc, Output, Input, callback, Dash, State
import dash_bootstrap_components as dbc


@callback(
    Output(CONFIG.store, 'data'),
    Input(CONFIG.next, 'n_clicks'),
    State(CONFIG.store, 'data'),
    prevent_initial_call=True)
def table_next_callback(next, mem):
    mem = CONFIG(mem)
    return table_next(mem)


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
    State(CONFIG.store, 'data'),
    prevent_initial_call=True
)
def upload_callback(content, lang, text, label, mem):
    mem = CONFIG(mem)
    return upload(mem, content, lang=lang,
                  text=text, label=label)


def run():
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],
               suppress_callback_exceptions=True)

    app.layout = dbc.Container([dcc.Loading(children=dcc.Store(CONFIG.store),
                                            fullscreen=True),
                                dbc.Row(table_component()),
                                dbc.Row(download_component()),
                                dbc.Row(upload_component())])
    app.run_server(debug=True)


if __name__ == '__main__':
    run()