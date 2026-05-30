"""Streamlit demo (primary serving surface).

Pick a horizon, view the forecast against recent history, with the non-advice
disclaimer always on screen. Uses the pre-fitted model; it never refits here.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from market_forecast import __version__
from market_forecast.config import settings
from market_forecast.predict import DISCLAIMER, ForecastRequest, load_artifact, predict

st.set_page_config(page_title="market-forecast", page_icon=":chart_with_upwards_trend:")

st.title("Market forecasting demo")
st.caption(f"version {__version__}")
st.error(f"**NOT FINANCIAL ADVICE.** {DISCLAIMER}")


@st.cache_data(show_spinner=False)
def _recent_history(lookback: int = 120) -> pd.Series | None:
    from market_forecast.data import load_series

    try:
        return load_series().iloc[-lookback:]
    except Exception:
        return None


try:
    artifact = load_artifact()
except FileNotFoundError as exc:
    st.warning(str(exc))
    st.stop()

st.subheader(f"Series: {artifact['series_id']}")
col1, col2, col3 = st.columns(3)
col1.metric("Trained through", artifact["train_end"][:10])
col2.metric("Last price", f"{artifact['last_price']:,.2f}")
col3.metric("Walk-forward MAPE", f"{artifact['summary_metrics']['mape']:.2f}%")

horizon = st.slider("Forecast horizon (business days)", 1, 30, settings.horizon)
response = predict(ForecastRequest(horizon=horizon))

forecast = pd.Series(
    [p.forecast for p in response.points],
    index=pd.to_datetime([p.date for p in response.points]),
    name="forecast",
)

history = _recent_history()
chart = pd.DataFrame({"forecast": forecast})
if history is not None:
    chart = pd.concat([history.rename("history").to_frame(), chart], axis=1)
st.line_chart(chart)

st.dataframe(
    forecast.rename("forecast price").to_frame().style.format("{:,.2f}"),
    use_container_width=True,
)

with st.expander("Walk-forward validation metrics"):
    if settings.metrics_path.exists():
        metrics = json.loads(settings.metrics_path.read_text())
        st.write(f"{metrics['n_splits']} expanding folds of {metrics['test_size']} days each.")
        st.dataframe(pd.DataFrame(metrics["summary"]).T.style.format("{:.3f}"))
        st.caption("Skill vs naive (positive = better than the random-walk baseline):")
        st.json(metrics["skill_vs_naive"])
    else:
        st.info("Run `make train` to generate validation metrics.")
