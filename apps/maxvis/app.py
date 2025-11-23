from __future__ import annotations
from typing import List, Tuple
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, ctx
import plotly.graph_objects as go
from .sparsemax import softmax, sparsemax
DEFAULT_Z = np.array([1.0, 0.0, -1.0], dtype=float)
SLIDER_MIN = -5.0
SLIDER_MAX = 5.0
SLIDER_STEP = 0.1

def make_simplex_mesh() -> List[go.BaseTraceType]:
    v1 = (1.0, 0.0, 0.0)
    v2 = (0.0, 1.0, 0.0)
    v3 = (0.0, 0.0, 1.0)
    xs = [v1[0], v2[0], v3[0]]
    ys = [v1[1], v2[1], v3[1]]
    zs = [v1[2], v2[2], v3[2]]
    mesh = go.Mesh3d(x=xs, y=ys, z=zs, i=[0], j=[1], k=[2], color='lightblue', opacity=0.25, name='Simplex', hoverinfo='skip', showscale=False, showlegend=False)
    edges = go.Scatter3d(x=[v1[0], v2[0], v3[0], v1[0]], y=[v1[1], v2[1], v3[1], v1[1]], z=[v1[2], v2[2], v3[2], v1[2]], mode='lines', line=dict(color='#888', width=5), name='Simplex edges', hoverinfo='skip', showlegend=False)
    return [mesh, edges]

def make_figure(z: np.ndarray) -> go.Figure:
    z = np.asarray(z, dtype=float)
    p_soft = softmax(z)
    p_sparse = sparsemax(z)
    min_v = float(min(0.0, np.min(z)))
    max_v = float(max(1.0, np.max(z)))
    span = max_v - min_v
    pad = 0.05 * span if span > 0 else 0.5
    axis_min = min_v - pad
    axis_max = max_v + pad
    traces = []
    traces.extend(make_simplex_mesh())
    axis_x = go.Scatter3d(x=[axis_min, axis_max], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='#000', width=3), name='axis p1', hoverinfo='skip', showlegend=False)
    axis_y = go.Scatter3d(x=[0, 0], y=[axis_min, axis_max], z=[0, 0], mode='lines', line=dict(color='#000', width=3), name='axis p2', hoverinfo='skip', showlegend=False)
    axis_z = go.Scatter3d(x=[0, 0], y=[0, 0], z=[axis_min, axis_max], mode='lines', line=dict(color='#000', width=3), name='axis p3', hoverinfo='skip', showlegend=False)
    traces.extend([axis_x, axis_y, axis_z])
    line = go.Scatter3d(x=[p_soft[0], p_sparse[0]], y=[p_soft[1], p_sparse[1]], z=[p_soft[2], p_sparse[2]], mode='lines', line=dict(color='#6a5acd', width=6), name='softmax ↔ sparsemax', showlegend=True, hovertemplate='<b>Δ path</b><extra></extra>')
    soft_pt = go.Scatter3d(x=[p_soft[0]], y=[p_soft[1]], z=[p_soft[2]], mode='markers', marker=dict(size=7, color='#1f77b4'), name='softmax(z)', hovertemplate='softmax(z)<br>p1=%{x:.3f}<br>p2=%{y:.3f}<br>p3=%{z:.3f}<extra></extra>')
    sparse_pt = go.Scatter3d(x=[p_sparse[0]], y=[p_sparse[1]], z=[p_sparse[2]], mode='markers', marker=dict(size=9, color='#d62728', symbol='diamond'), name='sparsemax(z)', hovertemplate='sparsemax(z)<br>p1=%{x:.3f}<br>p2=%{y:.3f}<br>p3=%{z:.3f}<extra></extra>')
    z_pt = go.Scatter3d(x=[z[0]], y=[z[1]], z=[z[2]], mode='markers', marker=dict(size=8, color='#000000', symbol='square-open'), name='z', hovertemplate='z (raw)<br>z1=%{x:.3f}<br>z2=%{y:.3f}<br>z3=%{z:.3f}<extra></extra>')
    traces.extend([line, soft_pt, sparse_pt, z_pt])
    label_pos = axis_max - 0.02 * (axis_max - axis_min)
    annotations = [dict(x=label_pos, y=0, z=0, text='z₁', showarrow=False, font=dict(family="'Computer Modern', 'Latin Modern Roman', 'Times New Roman', serif", size=16, color='#111')), dict(x=0, y=label_pos, z=0, text='z₂', showarrow=False, font=dict(family="'Computer Modern', 'Latin Modern Roman', 'Times New Roman', serif", size=16, color='#111')), dict(x=0, y=0, z=label_pos, text='z₃', showarrow=False, font=dict(family="'Computer Modern', 'Latin Modern Roman', 'Times New Roman', serif", size=16, color='#111'))]
    layout = go.Layout(margin=dict(l=0, r=0, t=0, b=0), scene=dict(xaxis=dict(title='', range=[axis_min, axis_max], backgroundcolor='#ffffff', showticklabels=False, ticks='', tickvals=[], ticktext=[], showgrid=False), yaxis=dict(title='', range=[axis_min, axis_max], backgroundcolor='#ffffff', showticklabels=False, ticks='', tickvals=[], ticktext=[], showgrid=False), zaxis=dict(title='', range=[axis_min, axis_max], backgroundcolor='#ffffff', showticklabels=False, ticks='', tickvals=[], ticktext=[], showgrid=False), aspectmode='cube', camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)), annotations=annotations), legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)', borderwidth=0), font=dict(family="'Computer Modern', 'Latin Modern Roman', 'Times New Roman', serif", size=14, color='#111'))
    return go.Figure(data=traces, layout=layout)

def fmt_float(v: float) -> str:
    return f'{v:.3f}'

def vector_row(label: str, vec: np.ndarray, highlight_zeros: bool=False) -> html.Div:
    items = []
    for (i, val) in enumerate(vec.tolist()):
        text = fmt_float(val)
        children: List = [text]
        if highlight_zeros and np.isclose(val, 0.0, atol=1e-12):
            children = [text, html.Span(' zero', style={'marginLeft': '6px', 'fontSize': '0.8em', 'color': '#b22222', 'backgroundColor': '#fdecec', 'padding': '1px 4px', 'borderRadius': '3px'})]
        items.append(html.Span(children))
        if i < 2:
            items.append(html.Span(', '))
    return html.Div([html.Span(label, style={'display': 'inline-block', 'minWidth': '100px', 'fontWeight': 600}), html.Span('['), *items, html.Span(']')], style={'marginBottom': '4px', 'fontFamily': 'monospace'})

def build_description() -> html.Div:
    return html.Div([html.P('Both outputs lie on the probability simplex x + y + z = 1.', style={'margin': '0 0 6px 0'}), html.P('Softmax is dense (strictly positive). Sparsemax is a projection that often yields exact zeros.', style={'margin': '0'})], style={'color': '#333', 'fontSize': '14px'})
app = Dash(__name__)
app.title = 'Softmax vs Sparsemax in 3D'
app.layout = html.Div([html.H1('Softmax vs Sparsemax in 3D', style={'marginBottom': '4px'}), build_description(), html.Hr(), html.Div([html.Div([html.H3('Controls'), html.Div('Adjust logits z = (z1, z2, z3)'), html.Label('z1'), dcc.Slider(id='z1-slider', min=SLIDER_MIN, max=SLIDER_MAX, step=SLIDER_STEP, value=float(DEFAULT_Z[0]), marks={}, tooltip={'always_visible': False}), html.Label('z2', style={'marginTop': '12px'}), dcc.Slider(id='z2-slider', min=SLIDER_MIN, max=SLIDER_MAX, step=SLIDER_STEP, value=float(DEFAULT_Z[1]), marks={}, tooltip={'always_visible': False}), html.Label('z3', style={'marginTop': '12px'}), dcc.Slider(id='z3-slider', min=SLIDER_MIN, max=SLIDER_MAX, step=SLIDER_STEP, value=float(DEFAULT_Z[2]), marks={}, tooltip={'always_visible': False}), html.Div([html.Button('Reset', id='reset-button', n_clicks=0, style={'marginRight': '8px'}), html.Button('Random logits', id='random-button', n_clicks=0)], style={'marginTop': '16px'})], style={'flex': '0 0 320px', 'paddingRight': '24px', 'borderRight': '1px solid #eee'}), html.Div([dcc.Graph(id='simplex-graph', figure=make_figure(DEFAULT_Z), style={'height': '520px'}), html.Div(id='prob-text', style={'marginTop': '8px'})], style={'flex': '1', 'paddingLeft': '24px'})], style={'display': 'flex', 'alignItems': 'flex-start'})], style={'maxWidth': '1080px', 'margin': '0 auto', 'padding': '16px'})

@app.callback(Output('z1-slider', 'value'), Output('z2-slider', 'value'), Output('z3-slider', 'value'), Input('reset-button', 'n_clicks'), Input('random-button', 'n_clicks'), State('z1-slider', 'value'), State('z2-slider', 'value'), State('z3-slider', 'value'))
def handle_buttons(reset_clicks: int, random_clicks: int, z1: float, z2: float, z3: float):
    trigger = ctx.triggered_id
    if trigger == 'reset-button':
        return (float(DEFAULT_Z[0]), float(DEFAULT_Z[1]), float(DEFAULT_Z[2]))
    if trigger == 'random-button':
        r = np.random.randn(3)
        return (float(r[0]), float(r[1]), float(r[2]))
    return (z1, z2, z3)

@app.callback(Output('simplex-graph', 'figure'), Output('prob-text', 'children'), Input('z1-slider', 'value'), Input('z2-slider', 'value'), Input('z3-slider', 'value'))
def update_outputs(z1: float, z2: float, z3: float):
    z = np.array([z1, z2, z3], dtype=float)
    p_soft = softmax(z)
    p_sparse = sparsemax(z)
    fig = make_figure(z)
    content = html.Div([html.H4('Values', style={'marginBottom': '6px'}), vector_row('logits z =', z), vector_row('softmax(z) =', p_soft), vector_row('sparsemax(z) =', p_sparse, highlight_zeros=True)])
    return (fig, content)

def main() -> None:
    app.run(debug=True)
if __name__ == '__main__':
    main()
