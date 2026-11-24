from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ArchiveSummary(BaseModel):
    id: str
    lecture_id: str
    course_name: str
    date: datetime
    main_points: List[str]
    qa_session: List[dict]
    assignments: List[str]

class AdaptiveMaterialOut(BaseModel):
    archive_id: str
    is_present: bool
    material_type: str
    content: str
    links: List[str]