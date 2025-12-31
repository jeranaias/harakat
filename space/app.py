"""
Harakat Arabic Diacritization - Hugging Face Spaces App
Provides both Gradio UI and FastAPI endpoints
"""

import gradio as gr
from harakat import diacritize

def diacritize_text(text):
    """Diacritize Arabic text"""
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

📊 **2.29% DER** • **1.53% DER (no case)** • **6 MB** • **73x smaller than SOTA**

📖 **99.997% Quran Accuracy** — Quranic text auto-detected

---

Paste any Arabic text and click Submit. Quranic phrases are automatically detected.

الصق أي نص عربي واضغط Submit. النصوص القرآنية تُكتشف تلقائياً.""",
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

# Also create a simple API endpoint
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class DiacritizeRequest(BaseModel):
    text: str

class DiacritizeResponse(BaseModel):
    success: bool
    input: str
    output: str
    error: str = None

@app.post("/api/diacritize")
async def api_diacritize(request: DiacritizeRequest):
    """API endpoint for diacritization"""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    if len(request.text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long (max 10000 chars)")

    try:
        result = diacritize(request.text.strip())
        return DiacritizeResponse(
            success=True,
            input=request.text,
            output=result
        )
    except Exception as e:
        return DiacritizeResponse(
            success=False,
            input=request.text,
            output="",
            error=str(e)
        )

@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "Harakat API", "version": "3.5"}

# Mount Gradio app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    demo.launch()
