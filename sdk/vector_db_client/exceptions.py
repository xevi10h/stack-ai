"""
SDK Exceptions
"""


class VectorDBAPIError(Exception):
    """Base exception for all Vector DB API errors"""

    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class VectorDBConnectionError(VectorDBAPIError):
    """Raised when connection to the API fails"""

    pass


class VectorDBNotFoundError(VectorDBAPIError):
    """Raised when a resource is not found (404)"""

    def __init__(self, resource_type: str, resource_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        message = f"{resource_type} with id {resource_id} not found"
        super().__init__(message, 404)


class VectorDBValidationError(VectorDBAPIError):
    """Raised when validation fails (422)"""

    pass


class VectorDBUnauthorizedError(VectorDBAPIError):
    """Raised when authentication fails (401)"""

    def __init__(self):
        super().__init__("Unauthorized: Invalid or missing API key", 401)


class VectorDBRateLimitError(VectorDBAPIError):
    """Raised when rate limit is exceeded (429)"""

    def __init__(self):
        super().__init__("Rate limit exceeded. Please try again later.", 429)
