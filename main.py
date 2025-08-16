# ----------------------------------------------------
# Unmasking the Minds of Transformers: 
# Probing Attention Head Specialization in BERT
# ----------------------------------------------------

# 🧩 1. Setup
!pip install transformers datasets bertviz torch matplotlib seaborn

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from bertviz import head_view

# ----------------------------------------------------
# 📥 2. Load Model & Tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, output_attentions=True)

# ----------------------------------------------------
# 📚 3. Load Dataset (SST-2 for demo)
dataset = load_dataset("glue", "sst2")
sample_text = dataset["validation"][0]["sentence"]

inputs = tokenizer(sample_text, return_tensors="pt")
labels = torch.tensor([dataset["validation"][0]["label"]]).unsqueeze(0)

# ----------------------------------------------------
# 🎯 4. Forward Pass with Attention Extraction
outputs = model(**inputs, labels=labels)
loss, logits, attentions = outputs.loss, outputs.logits, outputs.attentions

print("Sentence:", sample_text)
print("Predicted Label:", torch.argmax(logits, dim=1).item())

# ----------------------------------------------------
# 👀 5. Visualize Attention with BertViz
head_view(attentions, tokens=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

# ----------------------------------------------------
# 🔬 6. Head Importance via Masking
def mask_attention_heads(model, mask_dict):
    """
    mask_dict: {layer: [list of heads to zero]}
    """
    def hook(module, input, output):
        layer_idx = module.layer_idx
        if layer_idx in mask_dict:
            heads_to_mask = mask_dict[layer_idx]
            output = output.clone()
            output[:, heads_to_mask, :, :] = 0
        return output

    # Register hooks
    handles = []
    for i, layer in enumerate(model.bert.encoder.layer):
        handles.append(
            layer.attention.self.register_forward_hook(
                lambda m, inp, out, idx=i: mask_attention_heads(model, {i: [0]}))
        )
    return handles

# Example: Mask head 0 in layer 0
handles = mask_attention_heads(model, {0: [0]})
outputs_masked = model(**inputs, labels=labels)
print("Logits after masking:", outputs_masked.logits)

# Clean up hooks
for h in handles: h.remove()

# ----------------------------------------------------
# 📊 7. Task Evaluation (Optional Expansion)
# TODO: Run full SST-2 validation set with head masking
#       → Measure accuracy drop per head
