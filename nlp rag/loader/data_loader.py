from abc import ABC, abstractmethod

from get_embedding_function import get_embedding_function


class DataLoader(ABC):
    def __init__(self, chroma_path):
        self.chroma_path = chroma_path
        self.embedding_function = get_embedding_function()

    @abstractmethod
    def load_data(self):
        pass
