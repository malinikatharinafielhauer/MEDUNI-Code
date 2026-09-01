from datasets import load_dataset
import os 

DATASET_REPO = "intronhealth/afrimedqa_v2"
OUTPUT_CSV = "ophthalmology_questions.csv"

def retr_data():
    return load_dataset(DATASET_REPO)
   

def ophtho_filter_function(retrieved_obj):
    ophtho_rows = retrieved_obj ["train"].filter(lambda row: row["specialty"] == "Ophthalmology")
    print("Len of opthalmology rows:", (len(ophtho_rows)))
    return ophtho_rows



if __name__ == "__main__":
    if (os.path.exists(OUTPUT_CSV)):
        print("csv available")
    else:
        print("csv not available")
        

        stored_data = retr_data()
        print(set(stored_data["train"]["question_type"]))

        generated_rows = ophtho_filter_function(stored_data)
        generated_rows.to_csv(OUTPUT_CSV)
        print(generated_rows)



