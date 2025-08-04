import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from detector import predict_mask

UPLOAD_FOLDER = "webapp/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            label, confidence = predict_mask(filepath)
            result = {
                "label": label,
                "confidence": confidence,
                "image_path": filepath.replace("webapp/static/", "/static/")
            }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
