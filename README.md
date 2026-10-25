# ED Workload Scheduling Lab (No Surge Lab)

## Run on Windows
Open PowerShell in this folder and run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Keep `simulation_events.csv` in the same folder as `app.py`, or upload it using the sidebar.

## Changes
- Removed the Surge Lab tab and all synthetic surge controls/references.
- Kept Overview, Scheduler Walkthrough, Compare Policies, and Method & Data.
- Made numeric-field conversion tolerant of text values such as descriptive pain entries.
- Clarified that the walkthrough's total waiting time is the final simulation result, while elapsed time is measured up to the selected tick.

## Limitations
This is a discrete-time CPU scheduling simulation. Historical events are source inputs, while job types, priorities, and burst lengths are project-defined assumptions. It does not reproduce a hospital's actual CPU workload.
