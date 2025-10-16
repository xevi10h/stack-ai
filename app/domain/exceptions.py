from uuid import UUID


class DomainException(Exception):
    pass


class EntityNotFoundError(DomainException):
    def __init__(self, entity_type: str, entity_id: UUID):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} with id {entity_id} not found")


class EntityAlreadyExistsError(DomainException):
    def __init__(self, entity_type: str, entity_id: UUID):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} with id {entity_id} already exists")


class LibraryNotIndexedError(DomainException):
    def __init__(self, library_id: UUID):
        self.library_id = library_id
        super().__init__(
            f"Library {library_id} is not indexed. Please index it before querying"
        )


class InvalidEmbeddingDimensionError(DomainException):
    def __init__(self, expected: int, actual: int):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Invalid embedding dimension. Expected {expected}, got {actual}"
        )


class EmptyLibraryError(DomainException):
    def __init__(self, library_id: UUID):
        self.library_id = library_id
        super().__init__(f"Library {library_id} is empty. Cannot perform query")


class InvalidMetadataFilterError(DomainException):
    def __init__(self, message: str):
        super().__init__(f"Invalid metadata filter: {message}")
