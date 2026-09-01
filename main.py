#retrieval script

#imports
import json
import pandas as pd


#the model

model_name = "Qwen/Qwen2.5-7B-instruct"
adapter = "QLoRA"
CSV_PATH = "ophthalmology_questions.csv"


#load_data() / preprocess_data()

class Loader:

    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.preprocess_data()

    def preprocess_data(self):
        self.df["question"] = self.df["question"].str.lower().fillna("")
        self.df["answer_rationale_lower"] = self.df["answer_rationale"].str.lower().fillna("")

    def resolve_answer(self, row):
        if row["question_type"] == "mcq" and pd.notna(row["correct_answer"]):
            options = json.loads(row["answer_options"])
            return options.get(row["correct_answer"], row["correct_answer"])
        return row["answer_rationale"] if pd.notna(row["answer_rationale"]) else None

    def search(self, keyword):
        #returns (matches_with_answers, total_mentions_found)
        keyword = keyword.lower()
        mask = (
            self.df["question"].str.contains(keyword, na=False)
            | self.df["answer_rationale_lower"].str.contains(keyword, na=False)
        )
        mentions = self.df[mask]

        has_answer = mentions["correct_answer"].notna() | mentions["answer_rationale"].notna()
        answered = mentions[has_answer]

        return answered, len(mentions)


# __main__ interactive loop

if __name__ == "__main__":
    loader = Loader(data_path=CSV_PATH)

    while True:
        query = input("Enter a search term (or 'quit' to exit): ").strip()

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        results, total_mentions = loader.search(query)

        if total_mentions == 0:
            print(f"No mentions of '{query}' found at all. Try again.\n")
            continue

        if results.empty:
            print(f"Found {total_mentions} mention(s) of '{query}', "
                  f"but none have a usable answer in this dataset. Try again.\n")
            continue

        print(f"\nFound {len(results)} answered match(es) for '{query}' "
              f"(out of {total_mentions} total mentions):\n")
        for _, row in results.iterrows():
            print(f"Q: {row['question'].strip()}")
            print(f"A: {loader.resolve_answer(row)}")
            print()