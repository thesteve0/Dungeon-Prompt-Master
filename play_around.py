# try these embedders - https://www.sbert.net/examples/applications/semantic-search/README.html
# https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5

# use this to get a rough idea of the implications of context length
# https://wordcounter.net/words-per-page



from langchain_text_splitters import MarkdownHeaderTextSplitter, CharacterTextSplitter

rule_book = open(r"ChatDM/rules/Player_s Handbook.md")
lines = rule_book.read()
print("number of lines = " + str(len(lines)))


headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(lines)

from langchain_text_splitters import RecursiveCharacterTextSplitter

chunk_size = 250
chunk_overlap = 30
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

# Split
header_splits = text_splitter.split_documents(md_header_splits)
header_splits

char_text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

paragraph_splits = char_text_splitter.split_documents(md_header_splits)
paragraph_splits