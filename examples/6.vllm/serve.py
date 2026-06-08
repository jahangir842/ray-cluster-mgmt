import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/user/projects/vllm-deployment/vllm/models/3.1-8b-instruct"

app = FastAPI()

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print("Model ready.")


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = MODEL_PATH
    messages: List[Message]
    max_tokens: int = 200
    temperature: float = 0.0

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatResponse(BaseModel):
    model: str
    choices: List[Choice]


@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [m.model_dump() for m in request.messages]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    do_sample = request.temperature > 0
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=request.max_tokens,
            do_sample=do_sample,
            temperature=request.temperature if do_sample else None,
            top_p=0.9 if do_sample else None,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return ChatResponse(
        model=request.model,
        choices=[Choice(index=0, message=Message(role="assistant", content=text), finish_reason="stop")],
    )


@app.get("/v1/models")
def list_models():
    return {"data": [{"id": MODEL_PATH, "object": "model"}]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)