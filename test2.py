from transformers import pipeline

generator = pipeline("text-generation", model="chavinlo/gpt4-x-alpaca")

res = generator(
    "hello whats the weather like today",
    max_length="30",
    num_return_sequence=2
)

print(res)