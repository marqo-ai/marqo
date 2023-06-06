# import torch
# from open_clip import create_model_and_transforms, get_tokenizer
# import time
#
# model_name = "ViT-B-32"
# start = time.time()
# model, _, transforms = create_model_and_transforms(model_name=model_name, pretrained="openai", device="cuda")
# elapsed_time = time.time() - start
# print(f"PRE-WARMUP: It take {elapsed_time} seconds to load the model {model_name} in cuda")
#
# start = time.time()
# tokenizer = get_tokenizer(model_name=model_name)
# text_processed = tokenizer(["hello word"]).to("cuda")
# with torch.no_grad(), torch.cuda.amp.autocast():
#     _ = model.encode_text(text_processed)
# elapsed_time = time.time() - start
# print(f"PRE-WARMUP: It take {elapsed_time} seconds to encode_text with the model {model_name} in cuda")
# exit()

import torch
import time

# Start the timer
print(f"-----------------Pre-Warmup started--------------------")
start = time.time()

# Check if CUDA is available and select GPU if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a random tensor
x = torch.randn((1000, 1000), device=device)

# Do a simple CUDA operation
y = (x ** 2).sum().item()

# Measure the time
elapsed_time = time.time() - start
print(f"-----------------Pre-Warmup completed in {elapsed_time} seconds--------------------")

exit()
