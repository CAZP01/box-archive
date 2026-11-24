from fastapi import APIRouter, HTTPException, Status
from app.schemas.iot import RoomStatusUpdate, DeviceControl
from app.services.mqtt_client import mqtt_manager

router = APIRouter(prefix="/api/v1/iot", tags=["IoT & Class Automation"])

@router.post("/room-status", status_code=200)
async def update_room_status(payload: RoomStatusUpdate):
    """
    Menerima sinyal deteksi dari Face Recognition & Jadwal[cite: 1].
    Otomatis mengontrol AC, Lampu, Proyektor via MQTT[cite: 1].
    """
    if payload.is_active and payload.lecturer_detected:
        mqtt_manager.publish_device_control(payload.room_id, "AC", "ON")
        mqtt_manager.publish_device_control(payload.room_id, "LIGHTS", "ON")
        mqtt_manager.publish_device_control(payload.room_id, "PROJECTOR", "ON")
        return {"status": "success", "message": "Perangkat kelas otomatis diaktifkan, Perekaman dimulai."}
    else:
        mqtt_manager.publish_device_control(payload.room_id, "AC", "OFF")
        mqtt_manager.publish_device_control(payload.room_id, "LIGHTS", "OFF")
        mqtt_manager.publish_device_control(payload.room_id, "PROJECTOR", "OFF")
        return {"status": "success", "message": "Kelas selesai, Perangkat dimatikan."}

@router.post("/control-device")
async def manual_device_control(control: DeviceControl):
    mqtt_manager.publish_device_control(control.room_id, control.device_type, control.action)
    return {"status": "command_sent", "details": control}