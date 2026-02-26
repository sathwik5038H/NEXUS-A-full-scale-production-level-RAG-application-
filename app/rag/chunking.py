from langchain_text_splitters import RecursiveCharacterTextSplitter



class ChunkingService:

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):

        # We use tiktoken to count actual tokens instead of characters

        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(

            encoding_name="cl100k_base",

            chunk_size=chunk_size,

            chunk_overlap=chunk_overlap,

            separators=[

                "\n\n",

                "\n",

                r"\d+\.\s",

                ". ",

                " ",

                ""

            ],

    is_separator_regex=True

)



    def chunk_text(self, text: str) -> list[str]:

        """Splits raw text into semantically safe chunks."""

        return self.splitter.split_text(text)