import json
import sys

MIN_ACCURACY = 0.60
MIN_RECALL = 0.60


def main():
    with open("metrics.json", "r") as file:
        metrics = json.load(file)

    accuracy = metrics["accuracy"]
    recall = metrics["recall"]

    print("ML Quality Gate")
    print("---------------")
    print(f"Accuracy: {accuracy:.4f} (minimum required: {MIN_ACCURACY})")
    print(f"Recall  : {recall:.4f} (minimum required: {MIN_RECALL})")

    if accuracy < MIN_ACCURACY or recall < MIN_RECALL:
        print("QUALITY GATE FAILED")
        sys.exit(1)

    print("QUALITY GATE PASSED")


if __name__ == "__main__":
    main()
