import os
import sys
import argparse
import uvicorn
from fastapi import (
        FastAPI,
        HTTPException,
        Depends,
        status,
)
from fastapi.security import HTTPBearer
from typing import Optional, Union, List, Literal
from pydantic import BaseModel, Field, ValidationError
from argparse_pydantic import add_args_from_model, create_model_obj
from starlette.concurrency import run_in_threadpool
from sentence_transformers import SentenceTransformer

# Mapping from model name to hugging face repository
model_mapping = {
        'all-mpnet-base-v2': 'all-mpnet-base-v2',
        'multi-qa-mpnet-base-dot-v1': 'multi-qa-mpnet-base-dot-v1',
        'all-distilroberta-v1': 'all-distilroberta-v1',
        'all-MiniLM-L12-v2': 'all-MiniLM-L12-v2',
        'multi-qa-distilbert-cos-v1': 'multi-qa-distilbert-cos-v1',
        'all-MiniLM-L6-v2': 'all-MiniLM-L6-v2',
        'multi-qa-MiniLM-L6-cos-v1': 'multi-qa-MiniLM-L6-cos-v1',
        'paraphrase-multilingual-mpnet-base-v2': 'paraphrase-multilingual-mpnet-base-v2',
        'paraphrase-albert-small-v2': 'paraphrase-albert-small-v2',
        'paraphrase-multilingual-MiniLM-L12-v2': 'paraphrase-multilingual-MiniLM-L12-v2',
        'paraphrase-MiniLM-L3-v2': 'paraphrase-MiniLM-L3-v2',
        'distiluse-base-multilingual-cased-v1': 'distiluse-base-multilingual-cased-v1',
        'distiluse-base-multilingual-cased-v2': 'distiluse-base-multilingual-cased-v2',
        'multilingual-e5-small': 'intfloat/multilingual-e5-small',
        'multilingual-e5-base': 'intfloat/multilingual-e5-base',
        'multilingual-e5-large': 'intfloat/multilingual-e5-large',
}

class Settings(BaseModel):
    host: str = Field(default="localhost", description='Listen address')
    port: int = Field(default=8000, description='Listen port')
    ssl_keyfile: Optional[str] = Field(default=None, description='SSL key file for HTTPS')
    ssl_certfile: Optional[str] = Field(default=None, description='SSL certificate file for HTTPS')
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    cache_folder: Optional[str] = Field(default='./models', description='Cacheing directory')
    device: Optional[str] = Field(default=None, description='Set "cuda" or "cpu", if this is None, try to detect whether to use CUDA')


settings: Optional[Settings] = None
def set_settings(_settings: Settings):
    global settings
    settings = _settings

def get_settings():
    return settings

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(union_mode='left_to_right', description='Sentence to be embedded.') # Only accept string format
    model: str = Field(description='Model to be used embedded.')
    encoding_format: Optional[str] = Field(default='float', description='Now, this is ignored.', pattern=r"float") # Only accept float format
    user: Optional[str] = Field(None,description='Now, this is ignored.')

class Embedding(BaseModel):
    def __init__(self, _index:int, _embedding:List[float]):
        super().__init__()
        self.index = _index
        self.embedding = _embedding

    index: int = Field(default=0, description='')
    embedding: List[float] = Field(default=[], description='')
    object: Literal["embedding"] = Field(default='embedding', description='The object type, which is always "embedding"')

class EmbeddingUsage(BaseModel):
    prompt_tokens: int = Field(default=0, description='')
    total_tokens: int = Field(default=0, description='')

class EmbeddingResponse(BaseModel):
    object: Literal["list"] = Field(default='list')
    model: str = Field(default='', description='The model which is used for embedding.')
    data: List[Embedding] = Field(default=[], description='Embeddings')
    usage: EmbeddingUsage = Field(default=EmbeddingUsage(), description='No, this is no meanigs.')

bearer_scheme = HTTPBearer(auto_error=False)

app = FastAPI()

async def authenticate(
        settings: Settings = Depends(get_settings),
        authorization: Optional[str] = Depends(bearer_scheme),
):
    if settings.api_key is None:
        return True
    if authorization and authorization.credentials == settings.api_key:
        return authorization.credentials

    raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
    )

@app.post("/v1/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(authenticate)])
async def create_embeddings(request: EmbeddingRequest):
    return await create_embedding_impl(request)

async def create_embedding_impl(request: EmbeddingRequest) -> EmbeddingResponse:
    try:
        settings = get_settings()
        if False == (request.model in model_mapping):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported model is requrested.",) 
        model_name = model_mapping[request.model]

        model = SentenceTransformer(
                model_name_or_path=model_name,
                cache_folder=settings.cache_folder,
                device=settings.device)
        sentences = [request.input] if type(request.input) is str else request.input
        embeddings = model.encode(sentences)

        response = EmbeddingResponse()
        response.model = request.model
        response.usage = EmbeddingUsage()
        response.usage.prompt_tokens = 0
        response.usage.total_tokens = 0
        response.data = []
        index = 0
        for vector in embeddings:
            embedding = Embedding(index, vector.tolist())
            response.data.append(embedding)
            index = index+1
        return response
    except Exception as e:
        print(e, file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    add_args_from_model(parser, Settings)
    args = parser.parse_args()
    try:
        settings = create_model_obj(Settings, args)
    except Exception as e:
        print(e, file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    set_settings(settings)
    uvicorn.run(
            app,
            host=os.getenv('HOST', settings.host),
            port=os.getenv('POST', settings.port),
            ssl_keyfile=settings.ssl_keyfile,
            ssl_certfile=settings.ssl_certfile)

