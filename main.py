from train.train_erm import train_single_source_erm

if __name__ == "__main__":
    train_single_source_erm(
        dataset_name="pacs",
        source_domain="photo",
        num_classes=7,
        num_epochs=3,
        batch_size=64,
        learning_rate =1e-4
    )