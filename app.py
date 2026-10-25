from __future__ import annotations

import io
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="ED Workload Lab",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Visual styling -----------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: rgba(120,130,150,.09); border: 1px solid rgba(130,140,160,.20); padding: 14px 16px; border-radius: 14px;}
    div[data-testid="stTabs"] button {font-weight: 600;}
    .eyebrow {text-transform: uppercase; letter-spacing: .10em; font-size: .74rem; color: #6ba8ff; font-weight: 700;}
    .explain {border-left: 3px solid #4b91f1; padding: .65rem .9rem; background: rgba(75,145,241,.08); border-radius: 0 10px 10px 0;}
    .small-muted {font-size: .88rem; opacity: .78;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏥 Emergency Department Workload Scheduling Lab")
st.caption("A data-driven operating-systems experiment: ED events → software jobs → CPU scheduling → measurable queue behaviour.")

st.info(
    "**What this system does:** it uses historical ED visit/vital-sign events as workload input, creates concrete software-processing jobs, "
    "and simulates how FCFS, Priority Scheduling, and Round Robin share one CPU. It does **not** schedule patients for treatment or control a real hospital system."
)

DEFAULT_FILE = Path(__file__).with_name("simulation_events.csv")
BURST_PROFILE = {
    "Validate event": 2,
    "Normalize record": 2,
    "Persist event": 3,
    "Process vitals": 3,
    "Close visit": 1,
    "Generic event": 2,
}
EVENT_JOB_MAP = {
    "PATIENT_ARRIVAL": ["Validate event", "Normalize record", "Persist event"],
    "VITALS_UPDATE": ["Validate event", "Process vitals", "Persist event"],
    "VISIT_CLOSED": ["Validate event", "Close visit", "Persist event"],
}
JOB_PRIORITY = {
    "Validate event": 1,
    "Process vitals": 2,
    "Persist event": 3,
    "Normalize record": 3,
    "Close visit": 3,
    "Generic event": 4,
}

REQUIRED_COLUMNS = {"event_id", "stay_id", "event_time", "event_type"}

@st.cache_data(show_spinner=False)
def load_events(data_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data_bytes))
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(sorted(missing)))
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_time"]).copy()
    df["event_type"] = df["event_type"].fillna("UNKNOWN").astype(str)
    df["stay_id"] = df["stay_id"].astype(str)
    df["event_id"] = df["event_id"].astype(str)
    df = df.sort_values(["event_time", "event_id"], kind="stable").reset_index(drop=True)
    return df

def run_prototype_operations(event: dict[str, Any]) -> dict[str, Any]:
    """Concrete, lightweight operations used by this educational prototype."""
    event_type = str(event.get("event_type", "UNKNOWN"))
    stay_id = str(event.get("stay_id", ""))
    event_id = str(event.get("event_id", ""))
    if not event_id or not stay_id:
        raise ValueError("event_id and stay_id are required")
    normalized = {
        "event_id": event_id,
        "stay_id": stay_id,
        "event_type": event_type,
        "chiefcomplaint": str(event.get("chiefcomplaint", "") or ""),
    }
    numeric_fields = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]
    for field in numeric_fields:
        value = event.get(field)
        # Some source fields (notably pain/chief complaint) can contain text.
        # Preserve only valid numeric values in numeric fields; otherwise use None.
        try:
            normalized[field] = None if pd.isna(value) or str(value).strip() == "" else float(value)
        except (TypeError, ValueError):
            normalized[field] = None
    # A real serialization operation, used to demonstrate the processing payload.
    import json
    serialized = json.dumps(normalized, separators=(",", ":"), sort_keys=True)
    return {"normalized": normalized, "serialized": serialized}

def benchmark_operations(df: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
    """Benchmark actual prototype validation/normalization/serialization/SQLite insert work."""
    sample = df.head(sample_size).to_dict("records")
    if not sample:
        return pd.DataFrame(columns=["Operation", "Median time (µs)", "P95 time (µs)", "Samples"])
    timings: dict[str, list[float]] = {
        "Validate required identifiers": [],
        "Normalize event fields": [],
        "Serialize event payload": [],
        "SQLite insert transaction": [],
    }
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE events (
            event_id TEXT PRIMARY KEY, stay_id TEXT, event_type TEXT,
            event_time TEXT, payload TEXT
        )"""
    )
    import json
    for row in sample:
        start = time.perf_counter_ns()
        if not row.get("event_id") or not row.get("stay_id"):
            raise ValueError("Required event identifiers are missing")
        _valid = True
        timings["Validate required identifiers"].append((time.perf_counter_ns() - start) / 1000)

        start = time.perf_counter_ns()
        processed = run_prototype_operations(row)
        timings["Normalize event fields"].append((time.perf_counter_ns() - start) / 1000)

        start = time.perf_counter_ns()
        payload = json.dumps(processed["normalized"], separators=(",", ":"), sort_keys=True)
        timings["Serialize event payload"].append((time.perf_counter_ns() - start) / 1000)

        # SQLite insert is a concrete storage operation; duplicate IDs are avoided for repeated samples.
        start = time.perf_counter_ns()
        conn.execute(
            "INSERT OR REPLACE INTO events(event_id, stay_id, event_type, event_time, payload) VALUES (?, ?, ?, ?, ?)",
            (row["event_id"], row["stay_id"], row["event_type"], str(row["event_time"]), payload),
        )
        conn.commit()
        timings["SQLite insert transaction"].append((time.perf_counter_ns() - start) / 1000)
    conn.close()
    rows = []
    for name, vals in timings.items():
        s = pd.Series(vals, dtype=float)
        rows.append({
            "Operation": name,
            "Median time (µs)": round(float(s.median()), 2),
            "P95 time (µs)": round(float(s.quantile(.95)), 2),
            "Samples": len(vals),
        })
    return pd.DataFrame(rows)

def build_jobs(events: pd.DataFrame, interval: int, priority_mode: str) -> pd.DataFrame:
    """Convert selected historical events into modeled software-processing jobs."""
    source = events.copy().reset_index(drop=True)
    source["source_record"] = source["event_id"].astype(str)
    source["traffic_origin"] = "Historical source event"

    jobs = []
    for i, event in source.iterrows():
        # Source order is used for reproducibility; de-identified timestamps are metadata only.
        event_arrival = i * max(0, int(interval))
        event_type = str(event.get("event_type", "UNKNOWN"))
        chain = EVENT_JOB_MAP.get(
            event_type,
            ["Validate event", "Generic event", "Persist event"],
        )

        for stage, task_name in enumerate(chain):
            if priority_mode == "Prioritize validation":
                priority = 1 if task_name == "Validate event" else JOB_PRIORITY.get(task_name, 4)
            elif priority_mode == "Prioritize vital processing":
                priority = 1 if task_name == "Process vitals" else JOB_PRIORITY.get(task_name, 4)
            else:
                priority = JOB_PRIORITY.get(task_name, 4)

            jobs.append({
                "job_id": f"J{len(jobs) + 1:05d}",
                "event_id": str(event.get("event_id", "")),
                "source_record": str(event.get("source_record", event.get("event_id", ""))),
                "stay_id": str(event.get("stay_id", "")),
                "event_type": event_type,
                "task": task_name,
                "arrival": int(event_arrival),
                "burst": int(BURST_PROFILE[task_name]),
                "priority": int(priority),
                "stage": stage + 1,
                "traffic_origin": "Historical source event",
                "source_time": str(event.get("event_time", "")),
            })
    return pd.DataFrame(jobs)

def simulate(jobs: pd.DataFrame, algorithm: str, quantum: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Single-core discrete-time scheduler. Priority is non-preemptive; RR is preemptive."""
    if jobs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    future = jobs.sort_values(["arrival", "job_id"], kind="stable").to_dict("records")
    remaining = {j["job_id"]: int(j["burst"]) for j in future}
    first_start: dict[str, int] = {}
    finish: dict[str, int] = {}
    ready: list[dict[str, Any]] = []
    gantt: list[dict[str, Any]] = []
    queue_history: list[dict[str, Any]] = []
    current = None
    slice_used = 0
    t = 0
    idx = 0
    completed = 0
    total = len(future)
    safety_limit = max(10000, sum(remaining.values()) + max(j["arrival"] for j in future) + 1000)

    while completed < total and t < safety_limit:
        while idx < total and future[idx]["arrival"] <= t:
            ready.append(future[idx])
            idx += 1

        # Select a new job only when CPU is free.
        if current is None and ready:
            if algorithm == "Priority":
                ready.sort(key=lambda j: (j["priority"], j["arrival"], j["job_id"]))
            # FCFS and RR use FIFO order for the ready queue.
            current = ready.pop(0)
            first_start.setdefault(current["job_id"], t)
            slice_used = 0

        queue_history.append({
            "tick": t,
            "ready_jobs": len(ready),
            "running_job": current["job_id"] if current else "IDLE",
            "completed_jobs": completed,
            "arrived_jobs": idx,
            "backlog_jobs": max(0, idx - completed),
        })

        if current is None:
            gantt.append({"start": t, "end": t + 1, "job_id": "IDLE", "task": "CPU idle", "event_type": "—"})
            t += 1
            continue

        jid = current["job_id"]
        gantt.append({
            "start": t,
            "end": t + 1,
            "job_id": jid,
            "task": current["task"],
            "event_type": current["event_type"],
        })
        remaining[jid] -= 1
        t += 1

        if remaining[jid] == 0:
            finish[jid] = t
            completed += 1
            current = None
            slice_used = 0
        elif algorithm == "RR":
            slice_used += 1
            if slice_used >= max(1, int(quantum)):
                ready.append(current)
                current = None
                slice_used = 0
        # FCFS and non-preemptive Priority retain current until it completes.

    if completed != total:
        raise RuntimeError("Simulation stopped before all jobs completed; check workload settings.")

    rows = []
    lookup = {j["job_id"]: j for j in future}
    for jid, end_time in finish.items():
        j = lookup[jid]
        rows.append({
            **j,
            "first_start": first_start[jid],
            "finish": end_time,
            "waiting": max(0, end_time - j["arrival"] - j["burst"]),
            "response": first_start[jid] - j["arrival"],
            "turnaround": end_time - j["arrival"],
        })
    return pd.DataFrame(rows), pd.DataFrame(gantt), pd.DataFrame(queue_history)

def summarize(result: pd.DataFrame, gantt: pd.DataFrame, algorithm: str) -> dict[str, Any]:
    makespan = int(gantt["end"].max()) if not gantt.empty else 0
    busy = int((gantt["job_id"] != "IDLE").sum()) if not gantt.empty else 0
    if result.empty:
        return {}
    return {
        "Algorithm": algorithm,
        "Mean wait (ticks)": round(result["waiting"].mean(), 2),
        "P95 wait (ticks)": round(result["waiting"].quantile(.95), 2),
        "Mean first response (ticks)": round(result["response"].mean(), 2),
        "Mean completion time (ticks)": round(result["turnaround"].mean(), 2),
        "Throughput (jobs/tick)": round(len(result) / max(1, makespan), 3),
        "CPU busy (%)": round(100 * busy / max(1, makespan), 1),
        "Jobs completed": len(result),
    }

# ----------------------------- Sidebar controls -----------------------------
with st.sidebar:
    st.markdown("## Experiment setup")
    uploaded = st.file_uploader("Upload event CSV", type=["csv"])
    if uploaded:
        raw = uploaded.getvalue()
    elif DEFAULT_FILE.exists():
        raw = DEFAULT_FILE.read_bytes()
        st.caption("Using bundled historical event trace.")
    else:
        st.warning("Upload simulation_events.csv to begin.")
        st.stop()

    try:
        all_events = load_events(raw)
    except Exception as exc:
        st.error(f"Could not load events: {exc}")
        st.stop()

    st.caption(f"Loaded {len(all_events):,} valid event rows.")
    max_events = min(250, len(all_events))
    n_events = st.slider("Source events to use", min_value=min(10, len(all_events)), max_value=len(all_events), value=max_events, step=max(1, min(10, len(all_events)//20)))
    interval = st.slider(
        "Ticks between source events",
        min_value=0,
        max_value=5,
        value=1,
        help="0 releases all selected source events together; 1 releases one source event per tick.",
    )
    priority_mode = st.selectbox(
        "Software job priority policy",
        ["Default processing policy", "Prioritize validation", "Prioritize vital processing"],
        help="This is a software-processing experiment, not clinical triage priority.",
    )
    quantum = st.slider("Round Robin time quantum (ticks)", min_value=1, max_value=8, value=3)
    st.markdown("---")
    st.caption("All time values are simulated ticks, not seconds. One simulated CPU core is used for a fair comparison.")

selected_events = all_events.head(n_events).copy()
jobs = build_jobs(selected_events, interval, priority_mode)

# Process a sample through real prototype operations and benchmark the implementation.
try:
    benchmark = benchmark_operations(selected_events, sample_size=min(100, len(selected_events)))
except Exception as exc:
    benchmark = pd.DataFrame()
    st.warning(f"Prototype operation benchmark unavailable: {exc}")

results: dict[str, pd.DataFrame] = {}
gantt_by_algo: dict[str, pd.DataFrame] = {}
queues_by_algo: dict[str, pd.DataFrame] = {}
summaries = []
for algo in ["FCFS", "Priority", "Round Robin"]:
    result, gantt, qhist = simulate(jobs, algo, quantum=quantum)
    results[algo] = result
    gantt_by_algo[algo] = gantt
    queues_by_algo[algo] = qhist
    summaries.append(summarize(result, gantt, algo))
summary_df = pd.DataFrame(summaries)

# ----------------------------- Main navigation -----------------------------
tab_overview, tab_live, tab_compare, tab_method = st.tabs([
    "📍 Overview",
    "⚙️ Scheduler walkthrough",
    "📊 Compare policies",
    "🧪 Method & data"
])

with tab_overview:
    st.markdown('<div class="eyebrow">01 / Workload overview</div>', unsafe_allow_html=True)
    st.header("What work is entering the system?")
    event_count = len(selected_events)
    visits = selected_events["stay_id"].nunique()
    total_jobs = len(jobs)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Source events", f"{event_count:,}")
    m2.metric("ED visits represented", f"{visits:,}")
    m3.metric("Software jobs generated", f"{total_jobs:,}", help="Each source event expands into a small processing pipeline.")
    m4.metric("Simulated CPU cores", "1")

    st.markdown(
        '<div class="explain"><b>How to read this:</b> an ED event is not itself a CPU instruction. The prototype expands each event into modeled jobs such as validation, normalization and persistence. The OS algorithms schedule those jobs; they do not decide patient treatment.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Event types in the selected data")
        event_counts = selected_events["event_type"].value_counts().rename_axis("Event type").reset_index(name="Events")
        fig = px.bar(event_counts, x="Event type", y="Events", color="Event type", title="Historical source events", text="Events")
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10), xaxis_title="", yaxis_title="Number of source events")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("What software jobs are generated?")
        task_counts = jobs["task"].value_counts().rename_axis("Software job").reset_index(name="Jobs")
        fig = px.bar(task_counts, x="Jobs", y="Software job", orientation="h", color="Software job", title="Generated processing jobs", text="Jobs")
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10), xaxis_title="Jobs", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Example: one source event becomes software work")
    example = selected_events.iloc[0].to_dict()
    example_jobs = jobs[jobs["source_record"] == str(example.get("event_id", ""))].copy()
    c1, c2 = st.columns([1, 1.35])
    with c1:
        st.write("**Source event**")
        st.json({
            "event_id": str(example.get("event_id", "")),
            "stay_id": str(example.get("stay_id", "")),
            "event_type": str(example.get("event_type", "")),
            "recorded_time": str(example.get("event_time", "")),
        })
    with c2:
        st.write("**Jobs created by the prototype**")
        if example_jobs.empty:
            st.caption("No matching jobs found for this event.")
        else:
            st.dataframe(example_jobs[["job_id", "task", "arrival", "burst", "priority"]], hide_index=True, use_container_width=True)
    st.caption("Burst values and priority numbers are transparent simulation settings. They are not sourced from hospital CPU logs.")

with tab_live:
    st.markdown('<div class="eyebrow">02 / Scheduler walkthrough</div>', unsafe_allow_html=True)
    st.header("Watch jobs compete for the CPU")
    algo = st.selectbox("Scheduling policy to inspect", ["FCFS", "Priority", "Round Robin"], key="walk_algo")
    qhist = queues_by_algo[algo]
    gantt = gantt_by_algo[algo]
    result = results[algo]
    max_tick = int(gantt["end"].max()) if not gantt.empty else 0
    tick = st.slider("Move through simulated time", min_value=0, max_value=max(0, max_tick-1), value=0, key="walk_tick")
    current_row = qhist[qhist["tick"] == tick]
    if current_row.empty:
        current = "IDLE"; ready_count = 0; completed_count = 0; arrived_count = 0; backlog = 0
    else:
        row = current_row.iloc[0]
        current = row["running_job"]
        ready_count = int(row["ready_jobs"])
        completed_count = int(row["completed_jobs"])
        arrived_count = int(row["arrived_jobs"])
        backlog = int(row["backlog_jobs"])
    a, b, c, d = st.columns(4)
    a.metric("Simulated tick", tick)
    b.metric("CPU currently running", current)
    c.metric("Ready queue length", ready_count)
    d.metric("Jobs not yet complete", backlog)
    task_row = result[result["job_id"] == current] if current != "IDLE" else pd.DataFrame()
    if not task_row.empty:
        tr = task_row.iloc[0]
        st.markdown(
            f'<div class="explain"><b>CPU decision:</b> {current} — <b>{tr["task"]}</b> is running. '
            f'It arrived at tick {int(tr["arrival"])} with {int(tr["burst"])} ticks of modeled CPU work. '
            f'Its first CPU start was tick {int(tr["first_start"])} and its completion is tick {int(tr["finish"])}.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="explain">The CPU is idle at this tick because no ready job is available.</div>', unsafe_allow_html=True)

    st.subheader("CPU timeline")
    visible = gantt[gantt["start"] <= tick].tail(30).copy()
    if not visible.empty:
        fig = go.Figure()
        for _, r in visible.iterrows():
            label = r["job_id"] if r["job_id"] == "IDLE" else f'{r["job_id"]} · {r["task"]}'
            fig.add_trace(go.Bar(
                x=[r["end"] - r["start"]], y=["CPU"], base=[r["start"]],
                orientation="h", name=label, text=[label], textposition="inside",
                hovertemplate=f'{label}<br>tick {r["start"]} to {r["end"]}<extra></extra>',
                showlegend=False,
            ))
        fig.update_layout(barmode="overlay", height=180, margin=dict(l=15, r=15, t=15, b=20), xaxis_title="Simulated time (ticks)", yaxis_title="", xaxis=dict(range=[max(0, tick-25), max(5, tick+1)]))
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Ready queue at this moment")
    st.caption("Elapsed time is measured up to the selected tick. Total waiting is the final value from the completed simulation, not the wait accumulated so far.")
    # Reconstruct ready jobs as those arrived by tick, unfinished at tick, and not executing at tick.
    started_or_finished = result[(result["arrival"] <= tick) & (result["finish"] > tick)]
    running_id = current if current != "IDLE" else ""
    ready_ids = started_or_finished.loc[started_or_finished["job_id"] != running_id, "job_id"].tolist()
    ready_table = result[result["job_id"].isin(ready_ids)].sort_values(["arrival", "job_id"]).head(30)
    if ready_table.empty:
        st.caption("No other unfinished jobs are waiting at this tick.")
    else:
        ready_table = ready_table.copy()
        ready_table["total_waiting_after_simulation (ticks)"] = ready_table["waiting"]
        ready_table["elapsed_since_arrival (ticks)"] = (tick - ready_table["arrival"]).clip(lower=0)
        st.dataframe(
            ready_table[["job_id", "event_id", "task", "arrival", "burst", "priority",
                         "elapsed_since_arrival (ticks)", "total_waiting_after_simulation (ticks)"]],
            use_container_width=True,
            hide_index=True,
        )
    st.caption("Use the slider to move tick-by-tick. This is a deterministic playback of the completed simulation, not a connection to a live hospital.")


with tab_compare:
    st.markdown('<div class="eyebrow">03 / Compare policies</div>', unsafe_allow_html=True)
    st.header("Which scheduling policy behaves better for this workload?")
    st.write("All policies receive the same jobs generated from the selected historical event records. Lower waiting/response/turnaround is generally better; throughput and CPU busy percentage need workload context.")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    metric = st.selectbox("Metric to compare", ["Mean wait (ticks)", "P95 wait (ticks)", "Mean first response (ticks)", "Mean completion time (ticks)", "Throughput (jobs/tick)", "CPU busy (%)"])
    fig = px.bar(summary_df, x="Algorithm", y=metric, color="Algorithm", text=metric, title=f"{metric} across scheduling policies")
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10), xaxis_title="", yaxis_title=metric)
    st.plotly_chart(fig, use_container_width=True)

    best_low = metric in ["Mean wait (ticks)", "P95 wait (ticks)", "Mean first response (ticks)", "Mean completion time (ticks)"]
    winner = summary_df.sort_values(metric, ascending=best_low).iloc[0]
    st.markdown("### Plain-English interpretation")
    if best_low:
        st.success(f"For this specific model and workload, **{winner['Algorithm']}** has the lowest **{metric.lower()}** ({winner[metric]}). This does not mean it is universally best; changing event arrival intervals, burst assumptions, priority policy or RR quantum may change the result.")
    else:
        st.info(f"For this specific model and workload, **{winner['Algorithm']}** has the highest **{metric.lower()}** ({winner[metric]}). Higher is not automatically better for every metric: CPU busy time, for example, is not a quality score on its own.")
    st.subheader("Job-level results")
    chosen = st.selectbox("Inspect jobs scheduled by", ["FCFS", "Priority", "Round Robin"], key="job_table_algo")
    display_cols = ["job_id", "event_id", "stay_id", "event_type", "task", "arrival", "burst", "priority", "first_start", "finish", "waiting", "response", "turnaround", "traffic_origin"]
    st.dataframe(results[chosen][display_cols].head(300), use_container_width=True, hide_index=True)
    st.download_button(
        "Download job metrics CSV",
        data=results[chosen][display_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"{chosen.lower().replace(' ', '_')}_job_metrics.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download policy summary CSV",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="scheduling_policy_summary.csv",
        mime="text/csv",
    )

with tab_method:
    st.markdown('<div class="eyebrow">04 / Method and data</div>', unsafe_allow_html=True)
    st.header("What is real, what is modeled, and what is measured?")
    st.subheader("Historical source data")
    st.write(
        f"The selected source contains **{len(selected_events):,} events** representing **{selected_events['stay_id'].nunique():,} distinct ED visits**. "
        "Original timestamps are shown as source metadata, but simulation arrivals are generated from the configurable tick interval because the de-identified dates are not used as reliable elapsed simulation time."
    )
    st.subheader("Concrete prototype operations")
    st.write("The app performs and benchmarks these real operations on a sample: validate required identifiers, normalize event fields, serialize an event payload, and insert the payload into an in-memory SQLite database.")
    if not benchmark.empty:
        st.dataframe(benchmark, use_container_width=True, hide_index=True)
        st.caption("These are measurements of this laptop/runtime and the selected sample. They are not the original hospital system's CPU times, and they are not automatically treated as CPU burst lengths.")
    else:
        st.info("Benchmark data is unavailable for this dataset.")
    st.subheader("Scheduling model assumptions")
    assumptions = pd.DataFrame([
        {"Setting": "CPU cores", "Current model": "1", "Meaning": "One CPU timeline makes policy differences easier to compare."},
        {"Setting": "Burst profiles", "Current model": str(BURST_PROFILE), "Meaning": "Assumed CPU ticks per job type; clearly configurable in code."},
        {"Setting": "Priority", "Current model": priority_mode, "Meaning": "Software-processing policy, not clinical triage."},
        {"Setting": "Round Robin quantum", "Current model": str(quantum), "Meaning": "Maximum consecutive ticks before an unfinished job returns to the ready queue."},
    ])
    st.dataframe(assumptions, use_container_width=True, hide_index=True)
    st.subheader("Important limitations")
    st.markdown("""
    - This is a **simulation**, not a production hospital information system or a real-time OS.
    - Historical events are real source records; the software jobs and their CPU burst lengths are modeled by this project.
    - The original dataset does not provide hospital CPU burst times, scheduler traces, or actual application execution logs.
    - Database writes are benchmarked as prototype operations, but storage I/O is not modeled as a separate blocking device queue.
    - The app does not allocate beds, assign clinicians, diagnose patients, or decide treatment priority.
    """)

st.markdown("---")
st.caption("Emergency Department Workload Scheduling Using OS Algorithms · Educational prototype · One simulated CPU core · Model assumptions are exposed for inspection.")
