import unittest

from app import app

HIGH_RISK_EMPLOYEE = {
    "Age": 24,
    "MonthlyIncome": 2500,
    "YearsAtCompany": 1,
    "YearsInCurrentRole": 0,
    "JobSatisfaction": 1,
    "WorkLifeBalance": 1,
    "JobInvolvement": 1,
    "PerformanceRating": 3,
    "Overtime": "Yes",
    "BusinessTravel": "Frequently",
    "Department": "Sales",
    "MaritalStatus": "Single",
    "EducationLevel": "High School",
    "JobLevel": "Entry"
}

LOW_RISK_EMPLOYEE = {
    "Age": 50,
    "MonthlyIncome": 15000,
    "YearsAtCompany": 15,
    "YearsInCurrentRole": 8,
    "JobSatisfaction": 4,
    "WorkLifeBalance": 4,
    "JobInvolvement": 4,
    "PerformanceRating": 4,
    "Overtime": "No",
    "BusinessTravel": "Non-Travel",
    "Department": "Technology",
    "MaritalStatus": "Married",
    "EducationLevel": "Master",
    "JobLevel": "Senior"
}


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        for path in ["/", "/health"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["status"], "ok")

    def test_high_risk_prediction(self):
        response = self.client.post("/predict", json=HIGH_RISK_EMPLOYEE)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "LEAVE")

    def test_low_risk_prediction(self):
        response = self.client.post("/predict", json=LOW_RISK_EMPLOYEE)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "STAY")

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={"Age": 30, "MonthlyIncome": 5000}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())


if __name__ == "__main__":
    unittest.main()
