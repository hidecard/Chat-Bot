from transformers import GPT2LMHeadModel, GPT2Tokenizer

# GPT-2 model နဲ့ tokenizer ကို ယူပါ
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# မေးခွန်းတစ်ခု စမ်းမေးပါ
prompt = "Variable ဆိုတာ ဘာလဲ"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Chatbot: " + response)