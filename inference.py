import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
from transformers import ViTModel, AutoModelForCausalLM, AutoTokenizer

# ---------------------------
# Config
# ---------------------------
class InferenceConfig:
    vit_model = "google/vit-base-patch16-224"
    llm_model = "gpt2"
    checkpoint = "multitask_outputs/nanovlm_instruction_finetuned.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens = 24
    temperature = 0.7
    top_k = 50
    top_p = 0.9
    repetition_penalty = 1.15

config = InferenceConfig()

# ---------------------------
# Image Transform
# ---------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------------
# Model Definition (same as training)
# ---------------------------
class NanoVLM(nn.Module):
    def __init__(self, vit_model_name, llm_model_name):
        super().__init__()
        self.vision_encoder = ViTModel.from_pretrained(vit_model_name)
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float32
        )
        for p in self.llm.parameters():
            p.requires_grad = False

        vdim = self.vision_encoder.config.hidden_size
        tdim = self.llm.config.hidden_size
        self.vision_projection = nn.Sequential(
            nn.Linear(vdim, tdim*2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(tdim*2, tdim),
            nn.LayerNorm(tdim),
            nn.Dropout(0.05),
        )

    @torch.no_grad()
    def generate(self, pixel_values, input_ids):
        self.eval()

        v_out = self.vision_encoder(pixel_values=pixel_values)
        v_emb = self.vision_projection(v_out.last_hidden_state)

        generated = input_ids[0].tolist()

        for _ in range(config.max_new_tokens):
            cur_ids = torch.tensor([generated], device=config.device)
            t_emb = self.llm.get_input_embeddings()(cur_ids)

            combo = torch.cat([v_emb, t_emb], dim=1)
            logits = self.llm(inputs_embeds=combo).logits[0, -1, :]

            # repetition penalty
            for tok in set(generated):
                if logits[tok] < 0:
                    logits[tok] *= config.repetition_penalty
                else:
                    logits[tok] /= config.repetition_penalty

            logits = logits / config.temperature

            # top-k
            if config.top_k > 0:
                kth = torch.topk(logits, config.top_k)[0][..., -1, None]
                logits[logits < kth] = -float("inf")

            # softmax
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()

            if nxt == self.llm.config.eos_token_id:
                break

            generated.append(nxt)

        return torch.tensor([generated], device=config.device)


# ---------------------------
# Load model + tokenizer
# ---------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.llm_model)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model backbone...")
model = NanoVLM(config.vit_model, config.llm_model).to(config.device)

print("Loading fine-tuned weights...")
ckpt = torch.load(config.checkpoint, map_location=config.device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()


# ---------------------------
# Prediction function
# ---------------------------
@torch.no_grad()
def predict(image_path: str, question: str):
    img = Image.open(image_path).convert("RGB")
    pv = transform(img).unsqueeze(0).to(config.device)
    img.close()

    prompt = f"Instruction: {question}\nAnswer:"

    enc = tokenizer(prompt, return_tensors="pt").to(config.device)

    output_ids = model.generate(pv, enc.input_ids)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # extract only answer
    return full_text.split("Answer:")[-1].strip()


# Test locally before Streamlit
if __name__ == "__main__":
    test_img = "LLVIP/infrared/train/010001.jpg"
    test_q = "How many people are in this image?"
    print(predict(test_img, test_q))
