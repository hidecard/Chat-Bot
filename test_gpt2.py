from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import json

# Model နဲ့ Tokenizer ကို ယူပါ
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# ဒေတာဖတ်ပါ
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Dataset ပြင်ဆင်ပါ
class ChatDataset:
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        prompt = self.data[idx]["prompt"]
        response = self.data[idx]["response"]
        text = f"{prompt} [SEP] {response}"
        encoding = self.tokenizer(text, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }

dataset = ChatDataset(data, tokenizer)

# Training ပြင်ဆင်ပါ
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train စတင်ပါ
trainer.train()

# Model သိမ်းပါ
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")