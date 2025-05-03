from langchain.text_splitter import RecursiveCharacterTextSplitter

if __name__ == "__main__":
    r_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=10)
    print(
        r_splitter.split_text(
            "Hello, I'm Armin. How are you doing today? Hopefully nothing serious happened from yesterday!"
        )
    )
