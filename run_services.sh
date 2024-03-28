#!/bin/bash

# Start FastAPI app in the background
uvicorn api.app:app --host 0.0.0.0 --port 8000 &

# Start Streamlit app in the background
streamlit run /app/streamlit/streamlit_immo_eliza.py --server.port 8501 &


# Wait for any process to exit
wait -n

# Exit with the status of process that exited first
exit $?
