from .context_formatter import ContextFormatter
from .models import MinimalSource


# Mock a Chunk class (since you have one in chunking.py)
class MockChunk:
    def __init__(self, file_path, first, last, text):
        self.file_path = file_path
        self.first_character_index = first
        self.last_character_index = last
        self.text = text


# Create test data
sources = [
    MinimalSource(
        file_path="test.py", first_character_index=0, last_character_index=50
    ),
    MinimalSource(
        file_path="test2.py", first_character_index=100, last_character_index=150
    ),
]

chunks = [
    MockChunk("test.py", 0, 50, "print('Hello World!')"),
    MockChunk("test2.py", 100, 150, "def hello():\n    return 'world'"),
]

# Test
formatter = ContextFormatter(max_chunks=2, max_chunk_size=100)
result = formatter.format(sources, chunks)

print("Formatted Context:")
print(result)
