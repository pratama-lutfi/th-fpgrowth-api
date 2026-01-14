from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pandas as pd
import io
from services.market_basket import MarketBasketService

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Market Basket Analysis API")

# Configure CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    max_iter: int = Form(100),
    tabu_size: int = Form(18),
    k_focus_items: int = Form(50),
    tabu_threshold: float = Form(0.1),
    min_support: float = Form(0.001),
    min_confidence: float = Form(0.2)
):
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        # Read the file content
        contents = await file.read()
        
        # Load into DataFrame
        # Mirroring the ingestion code provided:
        # df = pd.read_csv('./market_basket_dataset.csv', sep=';', dtype={'BillNo': str}, low_memory=False)
        df = pd.read_csv(
            io.BytesIO(contents),
            sep=';',
            dtype={'BillNo': str},
            low_memory=False
        )
        
        # Basic Validation
        if 'BillNo' not in df.columns or 'Itemname' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail="CSV must contain 'BillNo' and 'Itemname' columns."
            )
            
        # Run Analysis
        results = MarketBasketService.analyze(
            df=df,
            max_iter=max_iter,
            tabu_size=tabu_size,
            k=k_focus_items,
            threshold=tabu_threshold,
            min_support=min_support,
            min_confidence=min_confidence
        )
        
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
