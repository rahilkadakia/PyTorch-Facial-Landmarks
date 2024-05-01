from flask import Flask, request
import torch
from process_image import process_image, Network
from flask import send_file
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app, resources={r"/process-image": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

###### Load Model only once, when Server is started ######

model = Network()
print("###### MODEL LOADED ######")
model.load_state_dict(torch.load("Models/face_landmarks.pth", map_location=torch.device('cpu')))
print("###### MODEL STATE DICT LOADED ######")
model.eval()

@app.route("/", methods=['GET', 'POST'])
@cross_origin(origin='*',headers=['Content-Type'])
def home():
    return "This API works!"

@app.route("/process-image", methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type'])
def POST_process_image():
    input_image = request.files['file'].read()
    buffer = process_image(model, input_image)
    return send_file(buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
