# Event structure inside runs (SYNTHETIC, exploratory, confirm seeds; experiments/run_event_structure.py)

| regime | mean events per run | mean terminal failure streak | pass rate, all events | pass rate, events before the terminal streak |
|---|---|---|---|---|
| stock | 7.5 | 5.00 | 0.318 | 0.967 |
| intermediate | 14.0 | 5.00 | 0.631 | 0.982 |
| evolved | 61.6 | 5.00 | 0.915 | 0.996 |

Pass rate by event index within a run (all events, 0-based index):

| regime | event 0 | events 1-2 | events 3-5 | events 6-11 | events 12+ |
|---|---|---|---|---|---|
| stock | 95/100 = 0.95 | 115/200 = 0.57 | 25/295 = 0.08 | 2/150 = 0.01 | n/a |
| intermediate | 100/100 = 1.00 | 200/200 = 1.00 | 273/300 = 0.91 | 256/548 = 0.47 | 52/249 = 0.21 |
| evolved | 100/100 = 1.00 | 200/200 = 1.00 | 300/300 = 1.00 | 600/600 = 1.00 | 4441/4962 = 0.90 |
