"""Domain exceptions and centralized error handling.

Idea: the rest of the code raises *domain* exceptions (e.g. FileTooLargeError)
without worrying about HTTP. Here we translate them into consistent JSON
responses. This keeps the business logic clean and the error format uniform.
"""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import ErrorResponse


class AppError(Exception):
    """Base application exception.

    `code` is a stable, machine-readable identifier (clients rely on it);
    `detail` is the human-readable message; `http_status` is the HTTP status to use.
    """

    code: str = "app_error"
    http_status: int = status.HTTP_400_BAD_REQUEST

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class InvalidFileTypeError(AppError):
    code = "invalid_file_type"
    http_status = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


class FileTooLargeError(AppError):
    code = "file_too_large"
    http_status = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


class EmptyFileError(AppError):
    code = "empty_file"
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY


class CorruptedFileError(AppError):
    code = "corrupted_file"
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY


class DocumentNotFoundError(AppError):
    code = "document_not_found"
    http_status = status.HTTP_404_NOT_FOUND


class InvalidDocumentIdError(AppError):
    code = "invalid_document_id"
    http_status = status.HTTP_400_BAD_REQUEST


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the exception handlers to the FastAPI app."""

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        payload = ErrorResponse(error=exc.code, detail=exc.detail)
        return JSONResponse(status_code=exc.http_status, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # E.g. the `file` field is missing entirely from the form.
        payload = ErrorResponse(
            error="validation_error",
            detail="Invalid request: " + "; ".join(e["msg"] for e in exc.errors()),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=payload.model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Safety net: any unexpected error becomes a clean 500,
        # without exposing the stack trace to the user.
        payload = ErrorResponse(
            error="internal_error",
            detail="An internal error occurred. Please try again later.",
        )
        return JSONResponse(status_code=500, content=payload.model_dump())
