from open_clip import create_model_and_transforms
import time

start = time.time()
model, _, transforms = create_model_and_transforms(model_name="ViT-L-14", pretrained="openai", device="cuda")
elapsed_time = time.time() - start
print("It take {} seconds to load the model ViT-L-14 in cuda".format(elapsed_time))