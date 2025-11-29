from typing import Tuple
import numpy as np
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# HardConcrete parameters
BETA = 0.1
GAMMA = -0.1
ZETA = 1.1
N_GRID = 500


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def logit(x: np.ndarray) -> np.ndarray:
    return np.log(x) - np.log1p(-x)


def s_cdf(y: np.ndarray, log_alpha: float) -> np.ndarray:
    """CDF of S at y in (0,1). F_S(y) = sigmoid(β logit(y) - log α)."""
    y = np.asarray(y)
    # Handle bounds explicitly
    out = np.zeros_like(y, dtype=float)
    inside = (y > 0) & (y < 1)
    t = BETA * logit(y[inside]) - log_alpha
    out[inside] = sigmoid(t)
    out[y >= 1] = 1.0
    out[y <= 0] = 0.0
    return out


def s_pdf(y: np.ndarray, log_alpha: float) -> np.ndarray:
    """PDF of S at y in (0,1): f_S(y) = β * u*(1-u) / (y(1-y)), u = sigmoid(β logit(y) - log α)."""
    eps = 1e-8
    y = np.clip(np.asarray(y, dtype=float), eps, 1.0 - eps)
    u = sigmoid(BETA * logit(y) - log_alpha)
    return BETA * u * (1.0 - u) / (y * (1.0 - y))


def zbar_pdf(t: np.ndarray, log_alpha: float) -> np.ndarray:
    """PDF of Z̄ = γ + (ζ-γ) S via change of variables."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    mask = (t > GAMMA) & (t < ZETA)
    if np.any(mask):
        s = (t[mask] - GAMMA) / (ZETA - GAMMA)
        out[mask] = s_pdf(s, log_alpha) / (ZETA - GAMMA)
    return out


def z_cont_pdf(t: np.ndarray, log_alpha: float) -> np.ndarray:
    """Continuous part of Z on (0,1): equals z̄ pdf restricted to [0,1]."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    mask = (t > 0.0) & (t < 1.0)
    if np.any(mask):
        out[mask] = zbar_pdf(t[mask], log_alpha)
    return out


def z_point_masses(log_alpha: float) -> Tuple[float, float]:
    """Return (P(Z=0), P(Z=1))."""
    s0 = (0.0 - GAMMA) / (ZETA - GAMMA)
    s1 = (1.0 - GAMMA) / (ZETA - GAMMA)
    # Clamp to [0,1]
    def Fs(val: float) -> float:
        if val <= 0.0:
            return 0.0
        if val >= 1.0:
            return 1.0
        return float(s_cdf(np.array([val]), log_alpha)[0])

    p0 = Fs(s0)
    p1 = 1.0 - Fs(s1)
    # Numerical safety
    p0 = float(np.clip(p0, 0.0, 1.0))
    p1 = float(np.clip(p1, 0.0, 1.0))
    return p0, p1


def make_figure(log_alpha: float) -> go.Figure:

    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=(
            "Base noise u ~ Uniform(0,1)",
            "Binary Concrete s",
            "Stretched \u0305z",
            "HardConcrete gate z",
        ),
        horizontal_spacing=0.07,
    )

    # Panel 1: u density (Uniform(0,1))
    x_u = np.linspace(0.0, 1.0, N_GRID)
    y_u = np.ones_like(x_u)
    fig.add_trace(go.Scatter(x=x_u, y=y_u, mode="lines", name="u density",
                              line=dict(color="#1f77b4", width=3), showlegend=False,
                              hovertemplate="u=%{x:.3f}<br>f=%{y:.3f}<extra></extra>"), row=1, col=1)

    # Panel 2: s density on (0,1)
    x_s = np.linspace(1e-6, 1.0 - 1e-6, N_GRID)
    y_s = s_pdf(x_s, log_alpha)
    fig.add_trace(go.Scatter(x=x_s, y=y_s, mode="lines", name="s density",
                              line=dict(color="#ff7f0e", width=3), showlegend=False,
                              hovertemplate="s=%{x:.3f}<br>f=%{y:.3f}<extra></extra>"), row=1, col=2)

    # Panel 3: z̄ density on (γ, ζ)
    x_zb = np.linspace(GAMMA, ZETA, N_GRID)
    y_zb = zbar_pdf(x_zb, log_alpha)
    fig.add_trace(go.Scatter(x=x_zb, y=y_zb, mode="lines", name="z̄ density",
                              line=dict(color="#2ca02c", width=3), showlegend=False,
                              hovertemplate="z̄=%{x:.3f}<br>f=%{y:.3f}<extra></extra>"), row=1, col=3)

    # Panel 4: z continuous density on (0,1) and point masses at 0 and 1
    eps = 1e-6
    x_z = np.linspace(eps, 1.0 - eps, N_GRID)
    y_z = z_cont_pdf(x_z, log_alpha)
    fig.add_trace(go.Scatter(x=x_z, y=y_z, mode="lines", name="z density (cont.)",
                              line=dict(color="#d62728", width=3), showlegend=False,
                              hovertemplate="z=%{x:.3f}<br>f=%{y:.3f}<extra></extra>"), row=1, col=4)

    p0, p1 = z_point_masses(log_alpha)
    y0 = float(z_cont_pdf(np.array([eps]), log_alpha)[0])
    y1 = float(z_cont_pdf(np.array([1.0 - eps]), log_alpha)[0])

    def pmass_marker(x: float, mass: float, label: str, color: str) -> go.Scatter:
        size_px = 4.0 + 60.0 * np.sqrt(max(mass, 0.0))
        # Place marker at the end of the continuous curve
        y_at = y0 if x <= 0.0 + 1e-12 else y1
        return go.Scatter(x=[x], y=[y_at], mode="markers",
                          marker=dict(size=size_px, color=color, line=dict(color="#333", width=1)),
                          name=label, showlegend=True,
                          hovertemplate=f"{label}: {mass:.3f}<extra></extra>")

    fig.add_trace(pmass_marker(0.0, p0, "P(z = 0)", "#9467bd"), row=1, col=4)
    fig.add_trace(pmass_marker(1.0, p1, "P(z = 1)", "#8c564b"), row=1, col=4)

    # Axes ranges
    fig.update_xaxes(range=[0, 1], row=1, col=1)
    fig.update_xaxes(range=[0, 1], row=1, col=2)
    fig.update_xaxes(range=[GAMMA, ZETA], row=1, col=3)
    fig.update_xaxes(range=[0, 1], row=1, col=4)
    fig.update_yaxes(rangemode='tozero', row=1, col=1)
    fig.update_yaxes(rangemode='tozero', row=1, col=2)
    fig.update_yaxes(rangemode='tozero', row=1, col=3)
    # For the z panel, include headroom for endpoint markers
    max_yz = float(max(np.max(y_z) if y_z.size else 0.0, y0, y1))
    fig.update_yaxes(range=[0.0, max_yz * 1.05 if max_yz > 0 else 1.0], row=1, col=4)

    fig.update_layout(
        title=f"HardConcrete distributions, log(alpha) = {log_alpha:.2f}",
        bargap=0.0,
        uirevision='keep-alpha',
        showlegend=True,
        margin=dict(l=20, r=20, t=44, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(
            family="'Computer Modern', 'Latin Modern Roman', 'Times New Roman', serif",
            size=14,
            color="#111",
        ),
    )
    return fig


def build_description() -> html.Div:
    return html.Div([
        html.P("True distributions at each stage (no sampling):", style={"margin": "0 0 6px 0"}),
        html.Ul([
            html.Li("u density on [0,1]"),
            html.Li("s density via exact transformation of Uniform → Binary Concrete"),
            html.Li("\u0305z density via affine stretch"),
            html.Li("z = clip(\u0305z, 0, 1) with continuous density on (0,1) and point masses at 0 and 1"),
        ], style={"margin": "6px 0"}),
    ], style={"color": "#333", "fontSize": "14px"})


app = Dash(__name__)
app.title = "HardConcrete Gate Distributions"

DEFAULT_LOG_ALPHA = 0.0

app.layout = html.Div([
    html.H1("HardConcrete Gate Distributions", style={"marginBottom": "4px"}),
    build_description(),
    html.Hr(),
    dcc.Graph(id="hist-graph", figure=make_figure(DEFAULT_LOG_ALPHA),
              style={"height": "min(26vw, 360px)"}),
    html.Div([
        html.Div([
            dcc.Slider(
                id="alpha-slider",
                min=-5.0,
                max=5.0,
                step=0.001,
                value=float(DEFAULT_LOG_ALPHA),
                marks={},
                tooltip={"always_visible": False},
                updatemode="drag",
                dots=False,
            ),
        ], style={"flex": "1"}),
        html.Span(id="alpha-text", style={"marginLeft": "12px", "fontFamily": "monospace", "minWidth": "64px", "display": "inline-block", "textAlign": "right"}),
    ], style={"marginTop": "12px", "display": "flex", "alignItems": "center"}),
], style={"maxWidth": "1080px", "margin": "0 auto", "padding": "16px"})


@app.callback(Output("hist-graph", "figure"), Output("alpha-text", "children"), Input("alpha-slider", "value"))
def update_figure(log_alpha: float):
    val = float(log_alpha)
    return make_figure(val), f"{val:.3f}"


def main() -> None:
    app.run(debug=True)


if __name__ == "__main__":
    main()
