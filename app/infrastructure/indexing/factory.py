from typing import Dict, Type

from app.infrastructure.indexing.base import VectorIndex
from app.infrastructure.indexing.flat_index import FlatIndex
from app.infrastructure.indexing.hnsw_index import HNSWIndex
from app.infrastructure.indexing.ivf_index import IVFIndex


class IndexFactory:
    _index_types: Dict[str, Type[VectorIndex]] = {
        "flat": FlatIndex,
        "hnsw": HNSWIndex,
        "ivf": IVFIndex,
    }

    @classmethod
    def create_index(cls, index_type: str) -> VectorIndex:
        if index_type not in cls._index_types:
            raise ValueError(
                f"Unknown index type: {index_type}. "
                f"Available types: {', '.join(cls._index_types.keys())}"
            )

        return cls._index_types[index_type]()

    @classmethod
    def get_available_types(cls) -> list[str]:
        return list(cls._index_types.keys())
