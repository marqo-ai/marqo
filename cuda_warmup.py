import torch
from open_clip import create_model_and_transforms, get_tokenizer
import time

model_name = "ViT-B-32"
start = time.time()
model, _, transforms = create_model_and_transforms(model_name=model_name, pretrained="openai", device="cuda")
elapsed_time = time.time() - start
print(f"PRE-WARMUP: It take {elapsed_time} seconds to load the model {model_name} in cuda")

start = time.time()
tokenizer = get_tokenizer(model_name=model_name)
text_processed = tokenizer(["hello word"]).to("cuda")
with torch.no_grad(), torch.cuda.amp.autocast():
    _ = model.encode_text(text_processed)
elapsed_time = time.time() - start
print(f"PRE-WARMUP: It take {elapsed_time} seconds to encode_text with the model {model_name} in cuda")
exit()