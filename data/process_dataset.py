from datasets import load_from_disk

def main():
    # Load the saved dataset from disk.
    dataset = load_from_disk("data/flickr30k_train")
    
    # Print the number of samples and a sample entry.
    print("Number of samples:", len(dataset))
    print("Sample entry:")
    print(dataset[0])
    
    # Optionally, inspect the column names to understand the structure.
    print("Columns in the dataset:", dataset.column_names)

if __name__ == "__main__":
    main()
