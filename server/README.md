### Steps to run the server

1. Install all dependencies
```bash
pip install -r requirements.txt
```

2. Create the datastore (time estimated: 4 minutes)
```bash
python -m app.data_ingestion
```

3. Run the server
```bash
uvicorn app.main:app --reload
```
