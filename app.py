import io
import json

from flask import Flask, request
import numpy as np
import cv2

app = Flask(__name__)
app.debug = True

import main as detector

@app.route('/')
def index():
    # image_file = request.files
    return 'Flask Working'

@app.post('/image')
def processImage():

    highest_results=-1
    highest_results_type=-1
    image_file = request.files['image']
    highest_results_type_class = {0:'checker',1:'plain',2:'stripped'}
    highest_results_cat_class = {0:'shirt',1:'t-shirt'}
    colorDetector = []
    threshold = 0.4

    image_file_read=image_file.read()
    nparr = np.frombuffer(image_file_read, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    for x1, y1, x2, y2, score, class_id in detector.check_Shirt_TShirt(img):
        if highest_results<class_id and score>threshold:
            highest_results=class_id
            colorDetector = x1, y1, x2, y2

    for x1, y1, x2, y2, score, class_id in detector.check_Type_Shirt(img):
        if highest_results_type < class_id and score>threshold:
            highest_results_type = class_id
            # colorDetector = x1, y1, x2, y2

    if highest_results in highest_results_cat_class:
        type=highest_results_cat_class[highest_results]
    else:
        type = 'Not Found'

    if highest_results_type in highest_results_type_class:
        pattern = highest_results_type_class[highest_results_type]
    else:
        pattern = 'Not Found'

    if highest_results in highest_results_cat_class:
        # x1, y1, x2, y2, score, class_id=detector.check_Shirt_TShirt(img)
        dominant=detector.getDominantColor(colorDetector,io.BytesIO(image_file_read))
    else:
        dominant=[]

    return {
        'dominant_color':dominant,
        'type':type,
        'pattern': pattern,
        # "dominant_color":detector.getDominantColor(x1,x2,y1,y2,io.BytesIO(image_file_read))
    }

if __name__ == '__main__':
    app.run(debug=True)