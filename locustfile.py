from locust import HttpUser, task, between
import json

class AbbreviationDetectionUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def detect_abbreviation(self):
        input_text = "NER EPI stem cell transplantation (SCT)"
        data = {"text": input_text}

        response = self.client.post("/predict", json=data)

        assert response.status_code == 200

        result = json.loads(response.text)
        predictions = result['predictions']
        print(predictions)

if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py --host http://0.0.0.0:5001")
