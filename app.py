import os, re, torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="AXIS Cloud Brain")

MODEL_ID = "Lucifer80/axis-cloud-1.7b"
print("Loading AXIS Cloud Brain from Hugging Face...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

class ChatRequest(BaseModel):
    message: str
    personality: str = "akai"

@app.post("/chat")
def chat(req: ChatRequest):
    sys_p = f"You are {req.personality.capitalize()}, the cloud intelligence neural core of AXIS.\nAddress the user as {'sir' if req.personality == 'akai' else 'boss'}.\nOutput strictly as:\n<action>FUNCTION_CALL</action>\n<reply>SPOKEN_REPLY</reply>"
    prompt = f"<|im_start|>system\n{sys_p}<|im_end|>\n<|im_start|>user\n{req.message}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=48, temperature=0.1, do_sample=False)
        
    gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    act_m = re.search(r'<action>(.*?)</action>', gen, re.DOTALL)
    rep_m = re.search(r'<reply>(.*?)</reply>', gen, re.DOTALL)
    return {
        "action": act_m.group(1).strip() if act_m else None,
        "reply": rep_m.group(1).strip() if rep_m else gen
    }

@app.get("/")
def health():
    return {"status": "online", "model": "AXIS-Cloud-1.7B"}
