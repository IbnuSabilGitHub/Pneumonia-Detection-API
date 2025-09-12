"""
Template untuk Endpoint
Ganti endpoint_name dengan nama endpoint yang sesuai (snake_case)
Ganti EndpointName dengan nama endpoint yang sesuai (PascalCase)
"""
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Depends
from ..docs.endpoint_name_metadata import EndpointNameMetadata  # Ganti endpoint_name dan EndpointName
from ..models.schemas import EndpointNameResponse, EndpointNameErrorResponse  # Ganti EndpointName
from ..core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Load metadata for endpoint documentation
endpoint_metadata = EndpointNameMetadata.get_metadata()  # Ganti EndpointName

# Template untuk GET endpoint sederhana
@router.get(
    "/endpoint-path",  # Ganti dengan path yang sesuai
    tags=["Tag Name"],  # Ganti dengan tag yang sesuai
    response_model=EndpointNameResponse,  # Ganti EndpointName
    **endpoint_metadata
)
async def get_endpoint_function() -> EndpointNameResponse:  # Ganti nama fungsi dan EndpointName
    """
    **🎯 [Endpoint Function Description]**
    
    [Detailed function description explaining what this endpoint does]
    
    **Returns:**
        EndpointNameResponse: [Description of return value]
        
    **Raises:**
        HTTPException: [Description of when exceptions are raised]
    """
    try:
        # Business logic here
        # Contoh: mendapatkan data dari service atau database
        result_data = {
            "key1": "value1",
            "key2": "value2"
        }
        
        return EndpointNameResponse(  # Ganti EndpointName
            status="success",
            timestamp=datetime.now().isoformat(),
            data=result_data,
            message="Operation completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to [action description]: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "error_code": "INTERNAL_ERROR",
                "timestamp": datetime.now().isoformat()
            }
        )


# Template untuk POST endpoint dengan input
@router.post(
    "/endpoint-path",  # Ganti dengan path yang sesuai
    tags=["Tag Name"],  # Ganti dengan tag yang sesuai
    response_model=EndpointNameResponse,  # Ganti EndpointName
    **endpoint_metadata
)
async def post_endpoint_function(
    # Parameter input - sesuaikan dengan kebutuhan
    input_data: dict  # Atau buat request model khusus
) -> EndpointNameResponse:  # Ganti EndpointName
    """
    **🎯 [POST Endpoint Function Description]**
    
    [Detailed function description for POST endpoint]
    
    **Args:**
        input_data: [Description of input parameter]
    
    **Returns:**
        EndpointNameResponse: [Description of return value]
        
    **Raises:**
        HTTPException: [Description of when exceptions are raised]
    """
    try:
        # Validasi input
        if not input_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Input data is required",
                    "error_code": "MISSING_INPUT",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Business logic untuk memproses input
        processed_data = process_input_data(input_data)
        
        return EndpointNameResponse(  # Ganti EndpointName
            status="success",
            timestamp=datetime.now().isoformat(),
            data=processed_data,
            message="Data processed successfully"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        logger.error(f"Validation error in [endpoint]: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(e),
                "error_code": "VALIDATION_ERROR",
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Failed to [action description]: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "error_code": "INTERNAL_ERROR",
                "timestamp": datetime.now().isoformat()
            }
        )


# Template untuk endpoint dengan file upload
from fastapi import UploadFile, File

@router.post(
    "/endpoint-path/upload",  # Ganti dengan path yang sesuai
    tags=["Tag Name"],  # Ganti dengan tag yang sesuai
    response_model=EndpointNameResponse,  # Ganti EndpointName
    **endpoint_metadata
)
async def upload_endpoint_function(
    file: UploadFile = File(..., description="File to upload")
) -> EndpointNameResponse:  # Ganti EndpointName
    """
    **🎯 [File Upload Endpoint Description]**
    
    [Description of file upload functionality]
    
    **Args:**
        file: Uploaded file (JPG, JPEG, PNG only, max 10MB)
    
    **Returns:**
        EndpointNameResponse: [Description of return value]
        
    **Raises:**
        HTTPException: Various errors for validation, file size, etc.
    """
    try:
        # Validasi file type
        if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": f"Invalid file type: {file.content_type}. Only JPG, JPEG, PNG are allowed",
                    "error_code": "INVALID_FILE_TYPE",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Validasi file size (contoh: 10MB)
        MAX_SIZE = 10 * 1024 * 1024  # 10MB in bytes
        file_content = await file.read()
        if len(file_content) > MAX_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error": f"File size exceeds limit of {MAX_SIZE / (1024*1024):.1f} MB",
                    "error_code": "FILE_TOO_LARGE",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Reset file pointer setelah membaca
        await file.seek(0)
        
        # Proses file
        processed_result = await process_uploaded_file(file)
        
        return EndpointNameResponse(  # Ganti EndpointName
            status="success",
            timestamp=datetime.now().isoformat(),
            data=processed_result,
            message="File processed successfully"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to process uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "File processing failed",
                "error_code": "PROCESSING_ERROR",
                "timestamp": datetime.now().isoformat()
            }
        )


# Template untuk endpoint dengan dependency injection
from ..core.dependencies import get_dependencies

@router.get(
    "/endpoint-path/with-deps",  # Ganti dengan path yang sesuai
    tags=["Tag Name"],  # Ganti dengan tag yang sesuai
    response_model=EndpointNameResponse,  # Ganti EndpointName
    **endpoint_metadata
)
async def endpoint_with_dependencies(
    deps = Depends(get_dependencies)
) -> EndpointNameResponse:  # Ganti EndpointName
    """
    **🎯 [Endpoint with Dependencies Description]**
    
    [Description of endpoint that uses dependency injection]
    
    **Returns:**
        EndpointNameResponse: [Description of return value]
    """
    try:
        # Menggunakan dependencies yang di-inject
        if not deps.is_initialized():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "Service not initialized",
                    "error_code": "SERVICE_NOT_READY",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Business logic menggunakan dependencies
        service_result = await deps.some_service.get_data()
        
        return EndpointNameResponse(  # Ganti EndpointName
            status="success",
            timestamp=datetime.now().isoformat(),
            data=service_result,
            message="Data retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "error_code": "INTERNAL_ERROR", 
                "timestamp": datetime.now().isoformat()
            }
        )


# Helper functions (sesuaikan dengan kebutuhan)
def process_input_data(data: dict) -> dict:
    """Process input data and return result."""
    # Implementasi logic pemrosesan data
    return {"processed": True, "original": data}


async def process_uploaded_file(file: UploadFile) -> dict:
    """Process uploaded file and return result."""
    # Implementasi logic pemrosesan file
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(await file.read()),
        "processed": True
    }


# Template untuk endpoint dengan rate limiting
from ..core.advanced_rate_limiting import rate_limit

@router.get(
    "/endpoint-path/limited",  # Ganti dengan path yang sesuai
    tags=["Tag Name"],  # Ganti dengan tag yang sesuai
    response_model=EndpointNameResponse,  # Ganti EndpointName
    dependencies=[Depends(rate_limit)],  # Rate limiting applied
    **endpoint_metadata
)
async def rate_limited_endpoint() -> EndpointNameResponse:  # Ganti EndpointName
    """
    **🎯 [Rate Limited Endpoint Description]**
    
    [Description of endpoint with rate limiting]
    
    **Rate Limiting:**
        - 5 requests per minute per IP
        - 100 requests per hour per IP
    
    **Returns:**
        EndpointNameResponse: [Description of return value]
    """
    try:
        # Business logic here
        result = {"data": "rate limited endpoint result"}
        
        return EndpointNameResponse(  # Ganti EndpointName
            status="success",
            timestamp=datetime.now().isoformat(),
            data=result,
            message="Rate limited operation completed"
        )
        
    except Exception as e:
        logger.error(f"Failed in rate limited endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "error_code": "INTERNAL_ERROR",
                "timestamp": datetime.now().isoformat()
            }
        )
