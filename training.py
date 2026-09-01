#training script
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset #from hufa 
from datasets import DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
    
)
import torch
from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parent



#paths are there for a practical reason. i would have to switch the model name every time if i loaded it directly
MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct"
CSV_PATH = "ophthalmology_questions.csv"
OUTPUT_DIR = "./ophthalmology_adapter" #where the model save pretrained (output.dir) writes just those small adapter weights) 



#dealing with the path & data 

def load_data():
    ds = load_dataset("csv", data_files = CSV_PATH)
    return ds

def missing_rationale(ds):
    ophth_variable = ds["train"].filter(lambda row: row["answer_rationale"] is None)
    print("Missing rows", len(ophth_variable))
    return ophth_variable


def preprocess_data(ds):
    ophtho = ds["train"]

    def is_useable(row):
        if row["question_type"] == "mcq":
            return row["correct_answer"] is not None
        else:
            return row["answer_rationale"] is not None
    ophtho = ophtho.filter(is_useable)


        

    def format_row(row):
        if row["question_type"] == "mcq":                                               #answer_rationale: is the explanation
            prompt = row["question"] + " Options: " + str(row["answer_options"])
            if row["answer_rationale"] is not None:
                response = row["correct_answer"] + " " + row["answer_rationale"]
            else:
                response = row["correct_answer"]


        elif row["question_type"] == "saq":
            prompt = row["question"]
            response = row["answer_rationale"]

        elif row["question_type"] == "consumer_queries":
            prompt = row["prompt"] + " " + row["question"]

            response = row["answer_rationale"]

        else:
            prompt = row["prompt"] + " " + row["question"]
            response = row["answer_rationale"]

        return {"prompt": prompt, "response": response}

    formatted = ophtho.map(format_row)
    return formatted

def tokenize_row(row, tokenizer, max_length=512):
    text = row["prompt"] + tokenizer.eos_token + row["response"]
    encoded = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded

def attach_lora(model):
            model = prepare_model_for_kbit_training(model)
            lora_config = LoraConfig(
                r=16,                       # TODO: tune
                lora_alpha=32,              # TODO: tune
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  
                lora_dropout=0.05,          # TODO: tune
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            return model


def bitsandbytes_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config

def load_model_and_tokenizer(model_repo=MODEL_REPO):
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    bnb_config = bitsandbytes_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        device_map="auto",
        quantization_config=bnb_config,  # use bnb_config here
    )
    return model, tokenizer
    
   


if __name__ == "__main__":
    ds = load_data() #load csv
    formatted = preprocess_data(ds) #runs the full filter and format pipeline
    print(len(formatted)) #actual proof we need right now. real row count after filtering
    for i in range(5):   #shows 5 real formated examples, to vatch previously broken rows
        print(formatted[i])
    missing_rationale(ds)

    model, tokenizer = load_model_and_tokenizer()
    model = attach_lora(model)

    tokenized = formatted.map(lambda row: tokenize_row(row, tokenizer))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,  # TODO: tune
        learning_rate=2e-4,  # TODO: tune
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )
    trainer.train()  #this is the actual training step, which will save the adapter weights to OUTPUT_DIR


