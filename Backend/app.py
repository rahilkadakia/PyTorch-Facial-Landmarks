from flask import Flask, request
import torch
from process_image import process_image, Network
from flask import send_file

app = Flask(__name__)

###### Load Model only once, when Server is started ######

model = Network()
print ("###### MODEL LOADED ######")
model.load_state_dict(torch.load("Models/model.pth", map_location=torch.device('cpu')))
print ("###### MODEL STATE DICT LOADED ######")
model.eval()

@app.route("/process-image", methods = ['POST'])
def POST_process_image():
    # file = request.files['image'].read()
    buffer = process_image(model)
    return send_file(buffer, mimetype='image/png')


if __name__ == '__main__':
    app.run(port=8000, debug=True)