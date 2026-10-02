To mount this framework in editable local development mode, run inside the text_anomaly_pkg/ directory:
bash
```
pip install -e .
```

You can then import it into any downstream inference script or streaming architecture
- python:
```
from text_anomaly_pkg import LiveTextInferenceRunner

runner = LiveTextInferenceRunner(model_path="generated/dueling_double_dqn_model.pt")
analysis = runner.process_stream_frame("April is the cruellest month breeding lilacs...")
print(f"Decoded Policy Action: {analysis['selected_action']}") 
```