from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    lecture_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    source_context: Optional[str] = None
    is_out_of_context: bool = False