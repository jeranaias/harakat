"""
Harakat Arabic Diacritization - Hugging Face Spaces App
Provides both Gradio UI and FastAPI endpoints with full OpenAPI documentation
"""

import gradio as gr
from fastapi import FastAPI, HTTPException, Query
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

# Import the diacritization function
from harakat import diacritize

# =============================================================================
# FastAPI App with OpenAPI Documentation
# =============================================================================

API_TITLE = "Harakat API"
API_VERSION = "3.5.1"
API_DESCRIPTION = """
# Harakat - High-Accuracy Arabic Diacritization API

**حَرَكَات** - Professional Arabic text diacritization (tashkeel) with state-of-the-art accuracy.

## Performance Metrics

| Metric | Value |
|--------|-------|
| Diacritic Error Rate (DER) | **2.29%** |
| DER (without case endings) | **1.53%** |
| Word Error Rate (WER) | **6.37%** |
| Quran Accuracy | **99.997%** |
| Model Size | **6.7 MB** |

## Features

- **Automatic Quran Detection**: Quranic phrases are detected and diacritized with near-perfect accuracy
- **Lightweight**: 62x smaller than comparable SOTA models
- **Fast**: Single-file inference, no external dependencies required
- **Production-Ready**: Handles batch processing and long texts efficiently

## Usage Examples

### Python
```python
import requests

response = requests.post(
    "https://jeranaias-harakat.hf.space/api/diacritize",
    json={"text": "بسم الله الرحمن الرحيم"}
)
print(response.json()["output"])
# بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ
```

### cURL
```bash
curl -X POST "https://jeranaias-harakat.hf.space/api/diacritize" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "العلم نور"}'
```

### JavaScript
```javascript
const response = await fetch("https://jeranaias-harakat.hf.space/api/diacritize", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text: "مرحبا بالعالم"})
});
const data = await response.json();
console.log(data.output);
```

## Links

- [GitHub Repository](https://github.com/jeranaias/harakat)
- [Documentation](https://jeranaias.github.io/harakat)
- [Google Colab Demo](https://colab.research.google.com/github/jeranaias/harakat/blob/main/examples/harakat_demo.ipynb)
- [PyPI Package](https://pypi.org/project/harakat)

---

**Author**: Jesse Morgan | **License**: MIT
"""

# Create FastAPI app with metadata
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    contact={
        "name": "Jesse Morgan",
        "url": "https://github.com/jeranaias/harakat",
        "email": "jeranaias@gmail.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Diacritization",
            "description": "Arabic text diacritization (tashkeel) endpoints"
        },
        {
            "name": "Health",
            "description": "Service health and status endpoints"
        },
        {
            "name": "Examples",
            "description": "Example texts for testing"
        }
    ]
)


# =============================================================================
# Pydantic Models with Documentation
# =============================================================================

class DiacritizeRequest(BaseModel):
    """Request model for diacritization"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Arabic text to diacritize (max 10,000 characters)",
        json_schema_extra={
            "example": "بسم الله الرحمن الرحيم"
        }
    )
    quran_mode: Optional[bool] = Field(
        default=None,
        description="Force Quran mode (True), disable it (False), or auto-detect (None/omit)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "الحمد لله رب العالمين",
                "quran_mode": None
            }
        }


class DiacritizeResponse(BaseModel):
    """Response model for diacritization"""
    success: bool = Field(
        ...,
        description="Whether the diacritization was successful"
    )
    input: str = Field(
        ...,
        description="The original input text"
    )
    output: str = Field(
        ...,
        description="The diacritized output text"
    )
    quran_detected: Optional[bool] = Field(
        default=None,
        description="Whether Quranic text was detected"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if success is False"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "input": "بسم الله الرحمن الرحيم",
                "output": "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
                "quran_detected": True,
                "error": None
            }
        }


class BatchDiacritizeRequest(BaseModel):
    """Request model for batch diacritization"""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of Arabic texts to diacritize (max 100 texts)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "العلم نور",
                    "من جد وجد",
                    "الصبر مفتاح الفرج"
                ]
            }
        }


class BatchDiacritizeResponse(BaseModel):
    """Response model for batch diacritization"""
    success: bool = Field(
        ...,
        description="Whether all diacritizations were successful"
    )
    results: List[DiacritizeResponse] = Field(
        ...,
        description="List of diacritization results"
    )
    total: int = Field(
        ...,
        description="Total number of texts processed"
    )


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    model_size: str = Field(..., description="Model size")
    metrics: dict = Field(..., description="Performance metrics")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "Harakat API",
                "version": "3.5.0",
                "model_size": "6.7 MB",
                "metrics": {
                    "der": "2.29%",
                    "der_no_case": "1.53%",
                    "wer": "6.37%",
                    "quran_accuracy": "99.997%"
                }
            }
        }


class ExampleText(BaseModel):
    """Example text for testing"""
    category: str = Field(..., description="Category of the example")
    arabic: str = Field(..., description="Arabic text without diacritics")
    diacritized: str = Field(..., description="Expected diacritized output")


# =============================================================================
# API Endpoints
# =============================================================================

@app.post(
    "/api/diacritize",
    response_model=DiacritizeResponse,
    tags=["Diacritization"],
    summary="Diacritize Arabic text",
    description="""
Add diacritical marks (tashkeel/harakat) to Arabic text.

**Features:**
- Automatic Quran detection for near-perfect accuracy on Quranic text
- Handles Modern Standard Arabic and Classical Arabic
- Preserves non-Arabic characters and punctuation

**Limits:**
- Maximum 10,000 characters per request
- For larger texts, use the batch endpoint or split into chunks
    """,
    responses={
        200: {
            "description": "Successfully diacritized text",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "input": "العلم نور والجهل ظلام",
                        "output": "الْعِلْمُ نُورٌ وَالْجَهْلُ ظَلَامٌ",
                        "quran_detected": False,
                        "error": None
                    }
                }
            }
        },
        400: {
            "description": "Invalid request",
            "content": {
                "application/json": {
                    "examples": {
                        "empty": {"value": {"detail": "No text provided"}},
                        "too_long": {"value": {"detail": "Text too long (max 10000 chars)"}}
                    }
                }
            }
        }
    }
)
async def api_diacritize(request: DiacritizeRequest):
    """
    Diacritize Arabic text with automatic Quran detection.

    - **text**: Arabic text to diacritize (required)
    - **quran_mode**: Optional override for Quran detection
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    if len(request.text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long (max 10000 chars)")

    try:
        # Check for Quran content in input
        quran_indicators = ["الله", "الرحمن", "الرحيم", "رب العالمين", "يوم الدين"]
        has_quran_indicator = any(ind in request.text for ind in quran_indicators)

        result = diacritize(
            request.text.strip(),
            quran_mode=request.quran_mode
        )

        return DiacritizeResponse(
            success=True,
            input=request.text,
            output=result,
            quran_detected=has_quran_indicator if request.quran_mode is None else request.quran_mode
        )
    except Exception as e:
        return DiacritizeResponse(
            success=False,
            input=request.text,
            output="",
            error=str(e)
        )


@app.post(
    "/api/diacritize/batch",
    response_model=BatchDiacritizeResponse,
    tags=["Diacritization"],
    summary="Batch diacritize multiple texts",
    description="""
Diacritize multiple Arabic texts in a single request.

**Limits:**
- Maximum 100 texts per request
- Each text limited to 10,000 characters
    """
)
async def api_batch_diacritize(request: BatchDiacritizeRequest):
    """
    Batch diacritize multiple Arabic texts.

    - **texts**: List of Arabic texts to diacritize (max 100)
    """
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Too many texts (max 100)")

    results = []
    all_success = True

    for text in request.texts:
        try:
            if len(text) > 10000:
                results.append(DiacritizeResponse(
                    success=False,
                    input=text[:100] + "..." if len(text) > 100 else text,
                    output="",
                    error="Text too long (max 10000 chars)"
                ))
                all_success = False
            else:
                result = diacritize(text.strip())
                results.append(DiacritizeResponse(
                    success=True,
                    input=text,
                    output=result
                ))
        except Exception as e:
            results.append(DiacritizeResponse(
                success=False,
                input=text,
                output="",
                error=str(e)
            ))
            all_success = False

    return BatchDiacritizeResponse(
        success=all_success,
        results=results,
        total=len(results)
    )


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Check API health",
    description="Returns service status, version, and performance metrics."
)
async def health():
    """
    Health check endpoint with service information and metrics.
    """
    return HealthResponse(
        status="healthy",
        service="Harakat API",
        version=API_VERSION,
        model_size="6.7 MB",
        metrics={
            "der": "2.29%",
            "der_no_case": "1.53%",
            "wer": "6.37%",
            "quran_accuracy": "99.997%"
        }
    )


@app.get(
    "/api/examples",
    response_model=List[ExampleText],
    tags=["Examples"],
    summary="Get example texts",
    description="Returns a list of example Arabic texts for testing the API."
)
async def get_examples():
    """
    Get example texts for testing the diacritization API.
    """
    return [
        ExampleText(
            category="Quran - Al-Fatiha",
            arabic="الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين",
            diacritized="الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ الرَّحْمَنِ الرَّحِيمِ مَالِكِ يَوْمِ الدِّينِ"
        ),
        ExampleText(
            category="Quran - Basmala",
            arabic="بسم الله الرحمن الرحيم",
            diacritized="بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"
        ),
        ExampleText(
            category="Proverb",
            arabic="العلم نور والجهل ظلام",
            diacritized="الْعِلْمُ نُورٌ وَالْجَهْلُ ظَلَامٌ"
        ),
        ExampleText(
            category="Proverb",
            arabic="من جد وجد ومن زرع حصد",
            diacritized="مَنْ جَدَّ وَجَدَ وَمَنْ زَرَعَ حَصَدَ"
        ),
        ExampleText(
            category="Modern Standard Arabic",
            arabic="ذهب الولد الى المدرسة ليتعلم العلوم المفيدة",
            diacritized="ذَهَبَ الْوَلَدُ إِلَى الْمَدْرَسَةِ لِيَتَعَلَّمَ الْعُلُومَ الْمُفِيدَةَ"
        ),
        ExampleText(
            category="Greeting",
            arabic="مرحبا بالعالم",
            diacritized="مَرْحَبًا بِالْعَالَمِ"
        )
    ]


# =============================================================================
# Gradio Interface
# =============================================================================

def diacritize_text(text):
    """Diacritize Arabic text for Gradio interface"""
    if not text or not text.strip():
        return ""

    try:
        result = diacritize(text.strip())
        return result
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
demo = gr.Interface(
    fn=diacritize_text,
    inputs=gr.Textbox(
        label="Arabic Text (بدون تشكيل)",
        placeholder="اكتب أو الصق نصاً عربياً هنا...",
        lines=5,
        rtl=True,
        elem_id="input-box"
    ),
    outputs=gr.Textbox(
        label="Diacritized Text (مُشَكَّل)",
        lines=5,
        rtl=True,
        elem_id="output-box"
    ),
    title="حَرَكَات - Harakat",
    description="""**High-Accuracy Arabic Diacritization** | **تشكيل عربي عالي الدقة**

📊 **2.29% DER** • **1.53% DER (no case)** • **6.7 MB** • **62x smaller than SOTA**

📖 **99.997% Quran Accuracy** — Quranic text auto-detected

---

Paste any Arabic text and click Submit. Quranic phrases are automatically detected.

الصق أي نص عربي واضغط Submit. النصوص القرآنية تُكتشف تلقائياً.

📚 **[API Documentation](/docs)** | **[GitHub](https://github.com/jeranaias/harakat)**""",
    examples=[
        ["الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين"],
        ["بسم الله الرحمن الرحيم"],
        ["قل هو الله احد الله الصمد لم يلد ولم يولد"],
        ["ذهب الولد الى المدرسة ليتعلم العلوم المفيدة"],
        ["العلم نور والجهل ظلام"],
        ["من جد وجد ومن زرع حصد"],
    ],
    theme=gr.themes.Soft(),
    css="""
    #input-box textarea, #output-box textarea {
        font-family: 'Amiri', 'Noto Naskh Arabic', serif !important;
        font-size: 24px !important;
        line-height: 2 !important;
    }
    #output-box textarea {
        color: #10b981 !important;
        font-weight: 600 !important;
    }
    """,
    allow_flagging="never"
)

# Mount Gradio app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    demo.launch()
