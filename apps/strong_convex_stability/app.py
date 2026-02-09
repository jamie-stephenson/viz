from __future__ import annotations

import numpy as np
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go


# Domain and styling
W_MIN = -20.0
W_MAX = 130.0
N_POINTS = 2000

BLUE = "#1f77b4"    # tab:blue
ORANGE = "#ff7f0e"  # tab:orange


def _curves(w: np.ndarray) -> dict[str, np.ndarray]:
    """Compute base losses and empirical risks for the toy ERM example."""
    ell_zi = np.abs(w - 0.0)
    ell_zp = np.abs(w - 100.0)
    ell_other = np.abs(w - 10.0)

    L_S = 0.5 * (ell_zi + ell_other)
    L_Si = 0.5 * (ell_zp + ell_other)
    return {
        "ell_zi": ell_zi,
        "ell_zp": ell_zp,
        "L_S": L_S,
        "L_Si": L_Si,
    }


def _layout_common(title: str, ylabel: str) -> dict:
    return dict(
        title=title,
        xaxis=dict(title="parameter w"),
        yaxis=dict(title=ylabel, rangemode="tozero", range=[0, 200]),
        legend=dict(x=1.02, y=1.0, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=40, r=100, t=40, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(
            family="'Computer Modern', 'Latin Modern Roman', 'Times New Roman', serif",
            size=14,
            color="#111",
        ),
    )


def make_unregularized_figure() -> go.Figure:
    w = np.linspace(W_MIN, W_MAX, N_POINTS)
    curves = _curves(w)

    # Deterministic choices inside flat argmin sets
    w_S = 0.0      # argmin of L_S in [0, 10]
    w_Si = 100.0   # argmin of L_Si in [10, 100]
    y_S = abs(w_S)
    y_Si = abs(w_Si)

    fig = go.Figure()
    # Base pointwise losses
    fig.add_trace(go.Scatter(x=w, y=curves["ell_zi"], name="ℓ(w, zᵢ)=|w|",
                             line=dict(color="gray", width=1.8), opacity=0.5))
    fig.add_trace(go.Scatter(x=w, y=curves["ell_zp"], name="ℓ(w, z′)=|w-100|",
                             line=dict(color="gray", width=1.8, dash="dash"), opacity=0.5))
    # Empirical risks
    fig.add_trace(go.Scatter(x=w, y=curves["L_S"],  name="L_S(w)", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=w, y=curves["L_Si"], name="L_{S^i}(w)", line=dict(width=3)))

    # Colored verticals at the chosen minimizers (pointwise loss values on zᵢ)
    if y_S > 0:
        fig.add_trace(go.Scatter(x=[w_S, w_S], y=[0, y_S], mode="lines",
                                 line=dict(color=BLUE, width=4), name="ℓ(A(S), zᵢ)"))
    fig.add_trace(go.Scatter(x=[w_S], y=[y_S], mode="markers",
                             marker=dict(color=BLUE, size=8), showlegend=False))

    fig.add_trace(go.Scatter(x=[w_Si, w_Si], y=[0, y_Si], mode="lines",
                             line=dict(color=ORANGE, width=4), name="ℓ(A(S^i), zᵢ)"))
    fig.add_trace(go.Scatter(x=[w_Si], y=[y_Si], mode="markers",
                             marker=dict(color=ORANGE, size=8), showlegend=False))

    # Delta annotation
    delta = y_Si - y_S
    fig.add_annotation(x=w_Si + 25, y=min(190, y_Si + 35),
                       ax=w_Si, ay=y_Si, xref="x", yref="y",
                       axref="x", ayref="y",
                       text=f"Δℓ(zᵢ) = {delta:.0f}", showarrow=True)

    fig.update_layout(**_layout_common(
        title="Convex ERM without ℓ₂ Regularization (unstable)",
        ylabel="loss / empirical risk",
    ))
    return fig


def make_regularized_figure(lambda_reg: float = 0.05) -> go.Figure:
    w = np.linspace(W_MIN, W_MAX, N_POINTS)
    curves = _curves(w)
    reg = lambda_reg * (w ** 2)

    L_S_reg = curves["L_S"] + reg
    L_Si_reg = curves["L_Si"] + reg

    # Approximate minimizers on the grid
    w_S_reg = float(w[int(np.argmin(L_S_reg))])
    w_Si_reg = float(w[int(np.argmin(L_Si_reg))])
    y_Sr = abs(w_S_reg)
    y_Sir = abs(w_Si_reg)
    delta = y_Sir - y_Sr

    fig = go.Figure()
    # Base pointwise losses
    fig.add_trace(go.Scatter(x=w, y=curves["ell_zi"], name="ℓ(w, zᵢ)=|w|",
                             line=dict(color="gray", width=1.8), opacity=0.5))
    fig.add_trace(go.Scatter(x=w, y=curves["ell_zp"], name="ℓ(w, z′)=|w-100|",
                             line=dict(color="gray", width=1.8, dash="dash"), opacity=0.5))
    # Regularized risks
    fig.add_trace(go.Scatter(x=w, y=L_S_reg,  name="L_S(w)+λ‖w‖²", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=w, y=L_Si_reg, name="L_{S^i}(w)+λ‖w‖²", line=dict(width=3)))

    # Minimizer guides
    fig.add_vline(x=w_S_reg, line_width=2, line_dash="dash", line_color="black", opacity=0.35)
    fig.add_vline(x=w_Si_reg, line_width=2, line_dash="dash", line_color="black", opacity=0.35)

    # Colored verticals at pointwise losses
    fig.add_trace(go.Scatter(x=[w_S_reg, w_S_reg], y=[0, y_Sr], mode="lines",
                             line=dict(color=BLUE, width=4), name="ℓ(A(S), zᵢ)"))
    fig.add_trace(go.Scatter(x=[w_S_reg], y=[y_Sr], mode="markers",
                             marker=dict(color=BLUE, size=8), showlegend=False))

    fig.add_trace(go.Scatter(x=[w_Si_reg, w_Si_reg], y=[0, y_Sir], mode="lines",
                             line=dict(color=ORANGE, width=4), name="ℓ(A(S^i), zᵢ)"))
    fig.add_trace(go.Scatter(x=[w_Si_reg], y=[y_Sir], mode="markers",
                             marker=dict(color=ORANGE, size=8), showlegend=False))

    # Delta annotation
    fig.add_annotation(x=w_Si_reg + 25, y=y_Sir + 25,
                       ax=w_Si_reg, ay=y_Sir, xref="x", yref="y",
                       axref="x", ayref="y",
                       text=f"Δℓ(zᵢ) = {delta:.2f}", showarrow=True)

    fig.update_layout(**_layout_common(
        title="Convex ERM with ℓ₂ Regularization (stable)",
        ylabel="loss / regularized risk",
    ))
    return fig


app = Dash(__name__)
app.title = "Convex ERM Stability"

DEFAULT_LAMBDA = 0.05

app.layout = html.Div([
    html.H1("Convex ERM Stability: Unregularized vs ℓ₂"),
    html.P("Toy example showing instability without ℓ₂ regularization and stability with it."
           " Colored verticals highlight pointwise losses at the selected minimizers."),
    html.Hr(),

    html.Div([
        html.Div([
            html.H3("Unregularized"),
            dcc.Graph(id="fig-unreg", figure=make_unregularized_figure(), style={"height": "420px"}),
        ], style={"flex": "1", "paddingRight": "16px"}),

        html.Div([
            html.H3("With ℓ₂ (λ slider)"),
            dcc.Graph(id="fig-reg", figure=make_regularized_figure(DEFAULT_LAMBDA), style={"height": "420px"}),
            html.Div([
                html.Label("λ (L2 regularization)"),
                dcc.Slider(id="lambda-slider", min=0.0, max=0.3, step=0.005, value=DEFAULT_LAMBDA,
                           marks={}, tooltip={"always_visible": False}),
            ], style={"marginTop": "8px"}),
        ], style={"flex": "1", "paddingLeft": "16px"}),
    ], style={"display": "flex", "alignItems": "flex-start"}),
], style={"maxWidth": "1080px", "margin": "0 auto", "padding": "16px"})


@app.callback(Output("fig-reg", "figure"), Input("lambda-slider", "value"))
def update_regularized(lambda_reg: float):
    return make_regularized_figure(float(lambda_reg))


def main() -> None:
    app.run(debug=True)


if __name__ == "__main__":
    main()
