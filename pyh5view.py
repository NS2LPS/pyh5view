# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import os
import h5py
import numpy as np
import json
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#__data_dir__ = r'Z:\Topolux\Data'
__data_dir__ = '/Users/jerome/Documents/Code/sandbox'

def listdir(thedir):
    if thedir and os.path.isdir(thedir):
        res = [ os.path.join(thedir, name) for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name)) ]
        res.sort(reverse=True)
        return res
    return []

def listfiles(thedir):
    if thedir and os.path.isdir(thedir):
        return [ os.path.join(thedir, name) for name in os.listdir(thedir) if os.path.isfile(os.path.join(thedir, name)) ]
    return []


app.layout = html.Div([
    html.H1(children='HDF5 Viewer'),
    html.Div([
    html.Label('Year'),
    dcc.Dropdown(
        id='dropdown-year',
        options = [{'label': os.path.basename(d), 'value': d} for d in listdir(__data_dir__)],
        value = listdir(__data_dir__)[0],
        multi=False,
        clearable=False,
    ),
    html.Label('Day'),
    dcc.Dropdown(
        id='dropdown-day',
        multi=False,
    ),
    ], style={'columnCount': 2}),
    dcc.RadioItems(
    id='radio-plottype',
    options=[
        {'label': 'Line plot', 'value': 'line'},
        {'label': '2D map', 'value': 'map'},
    ],
    value='line',
    labelStyle={'display': 'inline-block'}
    ),  
    html.Label('File'),
        dcc.Dropdown(
            id='dropdown-file',
            multi=True,
    ),
    html.Label('Dataset'),
    dcc.Dropdown(
        id='dropdown-dataset',
        multi=True
    ),
    html.Div([
    html.Label('X variable'),
    dcc.Dropdown(
        id='dropdown-x',
        multi=False
    ),
    html.Label('Y variable'),
    dcc.Dropdown(
        id='dropdown-y',
        multi=False
    ),
    html.Label('Map line index'),
    dcc.Dropdown(
        id='dropdown-z',
        multi=False
    ),
    ], style={'columnCount': 3}),
    dcc.Graph(id="main-graph"),
    dcc.ConfirmDialogProvider(
        children=html.Button(
            'Code',
        ),
        id='clipboard',
        message=''
    ),
    html.H3(id='attrs'),
    dash_table.DataTable(id='table'),
    html.H3('')
])

@app.callback(Output('dropdown-day', 'options'),
              Output('dropdown-day', 'value'),
              Input('dropdown-year', 'value'),
              )
def update_day(year_dir):
    options = [{'label': os.path.basename(d), 'value': d} for d in listdir(year_dir)]
    return options, None

@app.callback(Output('dropdown-file', 'options'),
              Output('dropdown-file', 'value'),
              Input('dropdown-day', 'value'),
              )
def update_file(day_dir):
    options = [{'label': os.path.basename(f), 'value': f} for f in listfiles(day_dir) if f.endswith('.h5')]
    return options, []

@app.callback(Output('dropdown-file', 'multi'),
              Output('dropdown-dataset', 'disabled'),
              Output('dropdown-file', 'value'),
              Output('dropdown-z', 'disabled'),
              Input('radio-plottype', 'value'),
              State('dropdown-file', 'value'),
              )
def update_plottype(val, prev_file):
    if val=='line':
        return True, False, [prev_file] if prev_file else [], True
    else:
        return False, True, prev_file[0] if prev_file else None, False

@app.callback(Output('dropdown-dataset', 'options'),
              Output('dropdown-dataset', 'value'),
              Input('dropdown-file', 'value'),
              State('dropdown-dataset', 'value'),
              State('radio-plottype', 'value')
              )
def update_dataset(files, prev_dataset, plot_type):
    if plot_type=='line':
        if files is None or len(files)==0 or files==[None]:
            return [], []
        file_dataset = []
        for f in files:    
            with h5py.File(f,'r') as h5file:
                file_dataset.extend([ (f,k) for k in h5file.keys() ])
        if len(files)==1:
            options = [{'label': ds, 'value': json.dumps((f,ds))} for f,ds in file_dataset]
        else:
            options = [{'label': '{0}:{1}'.format(os.path.splitext(os.path.basename(f))[0],ds), 'value': json.dumps((f,ds))} for f,ds in file_dataset]
        new_datasets = [ds['value'] for ds in options]
        value = [p for p in prev_dataset if p in new_datasets]
        return options, value
    else:
        return [], []


@app.callback(Output('dropdown-x','options'),
              Output('dropdown-y','options'),
              Output('dropdown-z','options'),
              Output('dropdown-x','value'), 
              Output('dropdown-y','value'),
              Output('dropdown-z','value'),
              Input('dropdown-dataset', 'value'),
              State('dropdown-x', 'value'),
              State('dropdown-y', 'value'),
              State('dropdown-z', 'value'),
              State('radio-plottype', 'value'),
              State('dropdown-file', 'value')
              )
def update_vars(file_dataset, prev_x, prev_y, prev_z, plot_type, file):
    if plot_type=='line':
        if file_dataset is None or len(file_dataset)==0:
            return [], [], [], None, None, None
        file_dataset = [json.loads(f) for f in file_dataset]
        v = []
        files = list(set([ f for f,ds in file_dataset]))
        for f1 in files:
            with h5py.File(f1,'r') as h5file:
                for f2,ds in file_dataset:
                    if f1==f2:
                        v.append(set(h5file[ds].dtype.names))
        if len(v)==0:
            return [], [], [], None, None, None
        v = set.intersection(*v)
        options = [{'label': x, 'value': x} for x in v]
        new_options = [o['value'] for o in options]
        value_x = prev_x if prev_x in new_options else None
        value_y = prev_y if prev_y in new_options else None
        return options, options, [], value_x, value_y, None
    else:
        if file is not None:
            with h5py.File(file,'r') as h5file:
                ds0 = list(h5file.keys())[0]
                options = [{'label': x, 'value': x} for x in h5file[ds0].dtype.names]
                map_options = [{'label': x, 'value': x} for x in h5file[ds0].attrs.keys()]
            new_options = [o['value'] for o in options]
            new_map_options = [o['value'] for o in map_options]
            value_x = prev_x if prev_x in new_options else None
            value_y = prev_y if prev_y in new_options else None
            value_z = prev_z if prev_z in new_map_options else None
            return options, options, map_options, value_x, value_y, value_z
        else:
            return [], [], [], None, None, None


@app.callback(Output('main-graph','figure'),
              Output('clipboard', 'message'),
              Input('dropdown-x', 'value'),
              Input('dropdown-y', 'value'),
              Input('dropdown-z', 'value'),
              State('dropdown-dataset', 'value'),
              State('radio-plottype', 'value'),
              State('dropdown-file', 'value')
              )
def update_fig(x, y, z, file_dataset, plot_type, file):
    if plot_type=='line':
        if file_dataset is None or len(file_dataset)==0 or x is None or y is None:
            return {'data':[{'x': [], 'y': [], 'type': 'line', 'name': None}] , 'layout' : {'title' : '' , 'xaxis' : {'title': ''}, 'yaxis' : { 'title':  ''}}}, ""
        file_dataset = [json.loads(f) for f in file_dataset]
        data = []
        files = list(set([ f for f,ds in file_dataset]))
        cmd_plot = []
        cmd = ""
        for f1 in files:
            with h5py.File(f1,'r') as h5file:
                cmd += f"with h5py.File(r'{f1}','r') as h5file:\n"
                for f2,ds in file_dataset:
                    if f1==f2:
                        name = ds if len(files)==1 else '{0} {1}'.format(os.path.splitext(os.path.basename(f1))[0],ds)
                        cmd_name = ds if len(files)==1 else '{0}_{1}'.format(os.path.splitext(os.path.basename(f1))[0],ds)
                        cmd += f"    {cmd_name}_{x} = array(h5file['{ds}']['{x}'])\n"
                        cmd += f"    {cmd_name}_{y} = array(h5file['{ds}']['{y}'])\n"
                        cmd_plot.append({'x': f"{cmd_name}_{x}", 'y': f"{cmd_name}_{y}", 'label': f"{name}"})
                        data.append( {'x': np.array(h5file[ds][x]), 'y': np.array(h5file[ds][y]), 'type': 'line', 'name': name} )
        cmd += "fig,ax = subplots()\n"
        for plot in cmd_plot:
            cmd += "ax.plot({x},{y},label='{label}')\n".format(**plot)
        title =  os.path.relpath(files[0],__data_dir__) + ' ' + ' '.join([os.path.basename(f) for f in files[1:]])
        #cmd += f"ax.set_xlabel('{x}')\nax.set_ylabel('{y}')\nlegend()\nax.set_title(r'{title}')\nfig.tight_layout()"
        cmd += f"ax.set_xlabel('{x}')\nax.set_ylabel('{y}')\nlegend()"
        return {'data':data , 'layout' : {'title' : title , 'xaxis' : {'title': x}, 'yaxis' : { 'title':  y}}}, cmd
    else:
        if file is None or x is None or y is None or z is None:
            return {'data':[{'x': [], 'y': [], 'type': 'line', 'name': None}] , 'layout' : {'title' : '' , 'xaxis' : {'title': ''}, 'yaxis' : { 'title':  ''}}}, ""
        else:
            with h5py.File(file,'r') as h5file:
                l = len(h5file)
                k = list(h5file.keys())
                ds0 = k[0]
                xdata = np.array(h5file[ds0][x])
                y_ = np.array(h5file[ds0][y])
                ydata = np.empty((l,len(y_)),dtype=y_.dtype)
                zdata = np.empty(l)
                for i in range(l):
                    ydata[i,:] = h5file[k[i]][y]
                    zdata[i] = h5file[k[i]].attrs[z]
            title =  os.path.relpath(file,__data_dir__) 
            fname = os.path.splitext(os.path.basename(file))[0]
            cmd = f"fig,ax = subplots()\n{fname}=loadh5('{file}')\nm=ax.pcolormesh({fname}.{x}[0],{fname}.{z},{fname}.{y})\n"
            cmd += f"ax.set_xlabel('{x}')\nax.set_ylabel('{z}')\ncolorbar(m, label='{y}')"
            fig = go.Figure(data=go.Heatmap(x=xdata, y=zdata, z=ydata, type = 'heatmap', colorscale = 'Viridis', colorbar={"title": y}))
            fig.update_layout(title=title , xaxis_title=x, yaxis_title=z)
            return fig, cmd




@app.callback(Output('table','columns'),
              Output('table','data'),
              Output('attrs','children'),
              Input('dropdown-dataset', 'value'),
              )
def update_table(file_dataset):
    if file_dataset is None or len(file_dataset)==0:
        return [], [], ''
    file_dataset = [json.loads(f) for f in file_dataset]
    data = []
    files = list(set([ f for f,ds in file_dataset]))
    for f1 in files:
        with h5py.File(f1,'r') as h5file:
            for f2,ds in file_dataset:
                if f1==f2:
                    name = ds if len(files)==1 else '{0}:{1}'.format(os.path.splitext(os.path.basename(f1))[0],ds)
                    attrs = {'id':name}
                    attrs.update(h5file[ds].attrs)
                    data.append(attrs)
    columns = ['id']
    other_columns = list(set([k for d in data for k in d.keys() if k!='id']))
    other_columns.sort()
    columns.extend( other_columns )
    columns = [ {'name':c,'id':c} for c in columns]
    return columns, data, 'Attributes'


if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server(debug=False, host='0.0.0.0', port=5012)
