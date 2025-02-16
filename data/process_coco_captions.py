from datasets import load_from_disk

def main():
    dataset = load_from_disk("data/coco_captions_train")
    print("Number of samples:", len(dataset))
    print("Sample entry:")
    print(dataset[0])
    print("Columns in the dataset:", dataset.column_names)

if __name__ == "__main__":
    main()
