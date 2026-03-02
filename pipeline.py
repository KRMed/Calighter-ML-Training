import subprocess
import sys

def pipeline():

    scripts = [
        "scripts/data_processing/database.py",
        "scripts/validation/create_error_report.py",
        "scripts/validation/validate_bio_data.py",
    ]

    for script in scripts:
        result = subprocess.run(f"Python {script}", shell=True)
        if result.returncode != 0:
            print(f"Error running {script}")
            break

if __name__ == "__main__":
    pipeline()