import uvicorn
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    print(df.head())

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8080, reload=True)