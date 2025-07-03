
#!/bin/bash
uvicorn 01webapp.main:app --host=0.0.0.0 --port=$PORT
