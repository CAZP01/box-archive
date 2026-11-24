import paho.mqtt.client as mqtt
import json
from app.config import settings

class MQTTManager:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        print(f"[MQTT] Connected with result code {rc}")
        self.client.subscribe("boxarchive/room/+/status")

    def _on_message(self, client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        print(f"[MQTT] Topic: {msg.topic} | Payload: {payload}")

    def start(self):
        try:
            self.client.connect(settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"[MQTT] Connection failed: {e}")

    def publish_device_control(self, room_id: str, device: str, action: str):
        topic = f"boxarchive/room/{room_id}/control"
        payload = json.dumps({"device": device, "action": action})
        self.client.publish(topic, payload)

mqtt_manager = MQTTManager()