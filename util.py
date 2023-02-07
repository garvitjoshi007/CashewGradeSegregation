import base64
from wavelet import w2d
import joblib
import json
import cv2
import numpy as np
import sklearn

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def load_saved_artifacts():
    print("Loading saved artifacts....start")
    global __class_name_to_number
    global __class_number_to_name

    with open('./Artifacts/class_dict.json',"r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./Artifacts/saved_model.pkl',"rb") as f:
            __model = joblib.load(f)
    print("Loading saved artifacts....done!")

def load_img():
    img = cv2.imread("IMG_6826.JPEG")
    return img


def classify_image(img,filepath=None):
    #img = np.array(img)
    img = get_cv2_image_from_base64_string(img)
    img = cv2.resize(img,(300, 300))
    img_transform = w2d(img,'db1',5)
    #img = get_b64()
    combined_img = np.vstack((img.reshape(300*300*3,1), img_transform.reshape(300*300,1)))
    len_image_array = 300*300*3 + 300*300
    final = combined_img.reshape(1,len_image_array).astype(float)

    #result = class_number_to_name(__model.predict(final)[0])
    #return result
    results = []
    results.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return results


def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

'''def convert_b64(img):
    with open(img,"r") as f:
        my_string = base64.b64encode(f.read())
    string_64 = my_string.decode('utf-8')
    return string_64

def get_b64():
    with open('b64.txt') as f:
        return f.read()

def convert_to_b64(img):
    with open(img, "rb") as image2string:
        img_string = base64.b64encode(image2string.read())
    return img_string'''

def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

if __name__=="__main__":
    img = load_img()
    load_saved_artifacts()
    print(classify_image(img,None))
    