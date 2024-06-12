from datasets.jetclass.jet import JetTag

def main():
    root = "../data/jetclass"
    dataset = JetTag(root)
    dataset.download()

if __name__ == "__main__":
    main()
    