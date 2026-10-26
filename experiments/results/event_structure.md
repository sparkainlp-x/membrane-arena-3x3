# Event structure inside runs (SYNTHETIC, exploratory, confirm seeds; experiments/run_event_structure.py)

| regime | mean events per run | mean terminal failure streak | pass rate, all events | pass rate, events before the terminal streak |
|---|---|---|---|---|
| stock | 7.6 | 5.00 | 0.317 | 0.938 |
| intermediate | 14.0 | 5.00 | 0.633 | 0.984 |
| evolved | 61.5 | 5.00 | 0.914 | 0.995 |

Pass rate by event index within a run (all events, 0-based index):

| regime | event 0 | events 1-2 | events 3-5 | events 6-11 | events 12+ |
|---|---|---|---|---|---|
| stock | 96/100 = 0.96 | 115/200 = 0.57 | 26/296 = 0.09 | 3/160 = 0.02 | n/a |
| intermediate | 100/100 = 1.00 | 199/200 = 0.99 | 272/300 = 0.91 | 270/551 = 0.49 | 47/251 = 0.19 |
| evolved | 100/100 = 1.00 | 200/200 = 1.00 | 300/300 = 1.00 | 600/600 = 1.00 | 4417/4946 = 0.89 |
