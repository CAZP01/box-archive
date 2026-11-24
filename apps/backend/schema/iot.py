from pydantic import BaseModel

class RoomStatusUpdate(BaseModel):
    room_id: str
    lecture_id: str
    is_active: bool
    lecturer_detected: bool

class DeviceControl(BaseModel):
    room_id: str
    device_type: str  # 'ac', 'lights', 'projector'
    action: str        # 'ON' or 'OFF'