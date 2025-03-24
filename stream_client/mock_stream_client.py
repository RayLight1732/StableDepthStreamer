from stream_client.serializable_data import SerializableData

class MockStreamClient:
    def connect(self) -> bool:
        pass

    def disconnect(self):
        pass

    def send_data(self, data: SerializableData):
        pass