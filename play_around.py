#### We are using the nomic embedding model to start
### https://docs.nomic.ai/reference/python-api/embeddings#local-inference
### Use it from Ollama
### https://ollama.com/library/nomic-embed-text
### And compare to instructor XL
### https://huggingface.co/hkunlp/instructor-xl

# try these embedders - https://www.sbert.net/examples/applications/semantic-search/README.html
# https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5
# use this to get a rough idea of the implications of context length
# https://wordcounter.net/words-per-page
from langchain_text_splitters import MarkdownHeaderTextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter
from nomic import embed
import psycopg
from pgvector.psycopg import register_vector


rule_book = open(r"ChatDM/rules/playhandbook_small.md")
lines = rule_book.read()
print("number of lines = " + str(len(lines)))


headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(lines)
token_lengths =[]

# Make 2 arrays. One to hold metadata the other to hold the text
texts = []
for doc in md_header_splits:
    texts.append(doc.page_content)


#https://docs.nomic.ai/atlas/models/text-embedding
output = embed.text(
    texts=texts,
    model='nomic-embed-text-v1.5',
    task_type='search_document',
    inference_mode='local',
)


## database time
DB_NAME = "dnd_rag"
vector_size = 768
table_name = "dnd_info"
conn = psycopg.connect("host=localhost user=postgres password='letmein'", autocommit=True)
cursor = conn.cursor()

cursor.execute("SELECT datname FROM pg_database;")

list_database = cursor.fetchall()

if (DB_NAME,) in list_database:
    cursor.execute(("DROP database "+ DB_NAME +" with (FORCE);"))
    cursor.execute("create database " + DB_NAME + ";")
else:
    cursor.execute("create database " + DB_NAME + ";")

# Now close the connection and switch DB
conn.close()

connect_string = f"host=localhost user=postgres password='letmein' dbname='{DB_NAME}'"

conn = psycopg.connect(connect_string,  autocommit=True)
conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
conn.execute('CREATE EXTENSION IF NOT EXISTS postgis')
conn.close()



connect_string = f"host=localhost user=postgres password='letmein' dbname='{DB_NAME}'"
conn = psycopg.connect(connect_string, autocommit=True)
register_vector(conn)
#register(conn)

conn.execute('DROP TABLE IF EXISTS %s' %  table_name)

# ID is autogenerated and all the other columns besides embedding are in the payload
conn.execute("""CREATE TABLE %s (id bigserial PRIMARY KEY, 
                            metadata text, 
                            content text,
                            embedding vector(%s))""" % (table_name, vector_size,))
conn.commit()
# Copy in spatial data ST_Point(location["lon"), location["lat"])

with conn.cursor().copy("COPY %s (metadata, content, embedding) FROM STDIN" % table_name) as copy:
    for i in range (0,len(md_header_splits)):

        copy.write_row([str(md_header_splits[i].metadata), texts[i], (str(output["embeddings"][i]))])
        print(type(output["embeddings"][i]))
        print("line: " + str(i))

print("creating HNSW index")
conn.execute("set maintenance_work_mem to '350MB'")
conn.execute("""CREATE INDEX idx_%s_hnsw ON %s USING hnsw  
                            (embedding vector_cosine_ops) WITH (m = 10, ef_construction = 40)""" % (table_name, table_name))
conn.commit()
conn.close()





# for text in md_header_splits:
#     token_lengths.append(len(text.page_content.split()))
#
# plt.hist(token_lengths, bins=15, color='skyblue', edgecolor='black')
#
# # Adding labels and title
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.title('Basic Histogram')
#
# plt.show()
print("done")
