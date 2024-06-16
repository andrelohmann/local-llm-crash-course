from ctransformers import AutoModelForCausalLM

model = "TheBloke/Llama-2-7B-Chat-GGUF"
# model = "zoltanctoth/orca_mini_3B-GGUF"
file = "llama-2-7b-chat.Q2_K.gguf"
# file = "orca-mini-3b.q4_0.gguf"
llm = AutoModelForCausalLM.from_pretrained(model, model_file=file)


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "You are an AI assistant that follows instruction extremely well. Help as much as you can. Give short answers."

    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    # prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"

    return prompt


question = "Which citiy is the capital of India?"

for i in llm(get_prompt(question), stream=True):
    print(i, end="", flush=True)
print()

question = "And which is the one of the united states?"

for i in llm(get_prompt(question), stream=True):
    print(i, end="", flush=True)
print()
