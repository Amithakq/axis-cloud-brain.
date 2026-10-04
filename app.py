import os, re
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from ctransformers import AutoModelForCausalLM

app = FastAPI(title="AXIS Cloud Brain")

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"  # Fits smoothly within Render free RAM
print("Loading model on Render...")
llm = AutoModelForCausalLM.from_pretrained(
    "Lucifer80/axis-cloud-1.7b",
    model_type="llama",
    lib="avx2"
)

class ChatRequest(BaseModel):
    message: str
    personality: str = "akai"

@app.post("/chat")
def chat(req: ChatRequest):
    sys_p = f"You are {req.personality.capitalize()}, the cloud intelligence neural core of AXIS.\nAddress the user as {'sir' if req.personality == 'akai' else 'boss'}.\nOutput strictly as:\n<action>FUNCTION_CALL</action>\n<reply>SPOKEN_REPLY</reply>"
    prompt = f"<|im_start|>system\n{sys_p}<|im_end|>\n<|im_start|>user\n{req.message}<|im_end|>\n<|im_start|>assistant\n"
    
    gen = llm(prompt, max_new_tokens=48, temperature=0.1, stop=["<|im_end|>"])
    act_m = re.search(r'<action>(.*?)</action>', gen, re.DOTALL)
    rep_m = re.search(r'<reply>(.*?)</reply>', gen, re.DOTALL)
    return {
        "action": act_m.group(1).strip() if act_m else None,
        "reply": rep_m.group(1).strip() if rep_m else gen
    }

@app.get("/")
def health():
    return {"status": "online", "service": "AXIS Cloud Brain"}
