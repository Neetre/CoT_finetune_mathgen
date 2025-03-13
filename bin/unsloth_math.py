# !pip install datasets
# !pip install unsloth
# !pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
# https://github.com/Neetre/CoT_finetune_mathgen.git

import os

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

from huggingface_hub import login
from transformers import TrainingArguments
from datasets import load_dataset

import wandb

hugging_face_token = os.getenv("HUGGINGFACE_TOKEN")
wnb_token = os.getenv("WANDB_TOKEN")

login(hugging_face_token)

wandb.login(key=wnb_token)
run = wandb.init(
    project='Fine-tune-DeepSeek-R1-Distill-Llama-8B on open-r1-OpenR1-Math-220k', 
    job_type="training", 
    anonymous="allow"
)

max_seq_length = 4096
dtype = None
load_in_4bit = True 


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=hugging_face_token,
)


train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a math tutor and you are helping a student with their homework.
Solve the given math problem step by step. Show your reasoning and provide the final answer.

### Problem:
{}

### Answer:
{}
</think>

{}"""


dataset = load_dataset("open-r1/OpenR1-Math-220k", "default", split = "train[0:800]",trust_remote_code=True, cache_dir="./dataset_cache")
print(dataset)
print(dataset[1])
# print(dataset[1]["generations"][0])

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a math tutor and you are helping a student with their homework.
Solve the given math problem step by step. Show your reasoning and provide the final answer.

### Problem:
{}

### Answer:
<think>{}"""


EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    inputs = examples["problem"]
    cots = examples["generations"]
    outputs = examples["solution"]
    
    texts = []
    
    
    for input, cot, output in zip(inputs, cots, outputs):  
        text = train_prompt_style.format(input, cot[0].split("</think>")[0], output) + EOS_TOKEN
        texts.append(text)

    return {
        "text": texts,
    }

dataset_finetune = dataset.map(formatting_prompts_func, batched = True)
print(dataset_finetune["text"][0])

model_lora = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=[
        "q_proj", 
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

trainer = SFTTrainer(
    model=model_lora,
    tokenizer=tokenizer,
    train_dataset=dataset_finetune,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=10,
    packing = False,

    args=TrainingArguments(
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407, 
        output_dir="outputs",
    ),
)


question = """Unified State Exam in Anchuria. (7-11 grades, 2 points). In Anchuria, the Unified State Exam is taking place.
The probability of guessing the correct answer to each question on the exam is 0.25. In 2011, to receive a certificate, one needed to answer correctly 3 questions out of 20.
In 2012, the School Administration of Anchuria decided that 3 questions were too few. Now, one needs to answer correctly 6 questions out of 40.
The question is, if one knows nothing and simply guesses the answers, in which year is the probability of obtaining an Anchurian certificate higher - in 2011 or in 2012?"""

inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model_lora.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=4096,
    use_cache=True,
)

response = tokenizer.batch_decode(outputs)

try:
    print(response[0].split("### Response:")[1])
except Exception as e:
    print(e)


trainer_stats = trainer.train()

question = """Unified State Exam in Anchuria. (7-11 grades, 2 points). In Anchuria, the Unified State Exam is taking place.
The probability of guessing the correct answer to each question on the exam is 0.25. In 2011, to receive a certificate, one needed to answer correctly 3 questions out of 20.
In 2012, the School Administration of Anchuria decided that 3 questions were too few. Now, one needs to answer correctly 6 questions out of 40.
The question is, if one knows nothing and simply guesses the answers, in which year is the probability of obtaining an Anchurian certificate higher - in 2011 or in 2012?"""

inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model_lora.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=4096,
    use_cache=True,
)

response = tokenizer.batch_decode(outputs)

try:
    print(response[0].split("### Response:")[1])
except Exception as e:
    print(e)


try:
    model.save_pretrained_merged("DeepSeek-R1-Distill-Llama-8B-OpenR1-Math", tokenizer, save_method="merged_16bit")
    model.push_to_hub_merged("Neetree/DeepSeek-R1-Distill-Llama-8B-OpenR1-Math", tokenizer, save_method="merged_16bit", token=hugging_face_token)
except Exception as e:
    print(f"Model saving failed with error: {e}")
    wandb.log({"Model Saving Error": str(e)})
    raise e

wandb.finish()

'''
ImportError: cannot import name '_unsloth_get_batch_samples' from 'unsloth_zoo.loss_utils' (/home/ubuntu/CoT_finetune_mathgen/.venv/lib/python3.10/site-packages/unsloth_zoo/loss_utils.py)
'''