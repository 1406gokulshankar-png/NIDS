
import os
import sys
print("Python version:", sys.version.split()[0])
print()

libraries = ["pandas", "numpy", "sklearn", "flask",
             "flask_sqlalchemy", "shap", "joblib", "matplotlib"]

for name in libraries:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown version")
        print(f"OK    {name} ({version})")
    except ImportError:
        print(f"FAIL  {name} is NOT installed")

print()
if os.path.isdir("data"):
    files = os.listdir("data")
    if files:
        print("Files found in data/ folder:")
        for f in files:
            print("   ", f)
    else:
        print("FAIL  data/ folder is empty. Put the NSL-KDD files in it.")
else:
    print("FAIL  data/ folder not found. Create it inside nids_project.")