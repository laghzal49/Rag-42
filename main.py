from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"


tokenzier = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenzier.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenzier([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
output_ids = generated_ids[0][
    len(model_inputs.input_ids[0]) :
].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenzier.decode(
    output_ids[:index], skip_special_tokens=True
).strip("\n")
content = tokenzier.decode(
    output_ids[index:], skip_special_tokens=True
).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
