from flask import Flask,request,jsonify,render_template,Response
import util
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)


# configure the SQLite database, relative to the app instance folder
'''app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///ipml.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"]= False

# create the extension
db = SQLAlchemy(app)
app.app_context().push()

# initialize the app with the extension
db.init_app(app)'''

'''class Img(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    img = db.Column(db.Text,unique=True,nullable=False)
    name = db.Column(db.Text,nullable=False)
    mimetype = db.Column(db.Text,nullable=False)'''

@app.route('/',methods=['GET', 'POST'])
def hello_world():
    return render_template("index.html")

@app.route('/classify_image',methods=['GET','POST'])
#@cross_origin()
def classify_image():
    image_data = request.form['image_data']
    #img = db.session.query(Img).order_by(Img.id.desc()).first()
    #image_data = img.img
    response = jsonify(util.classify_image(image_data))
    response.headers.add('Access-Control-Allow-Origin','*')
    return response


'''@app.route('/upload',methods=['POST'])
def upload():
    pic = request.files['pic']

    if not pic:
        return 'No picture uploaded!',400
    filename = secure_filename(pic.filename)
    mimetype = pic.mimetype

    img = Img(img=pic.read(),mimetype=mimetype,name=filename)
    db.session.add(img)
    db.session.commit()

    return 'Image has been uploaded',200

@app.route('/recent')
def get_img():
    img = db.session.query(Img).order_by(Img.id.desc()).first()
    if not img:
        return 'Img Not Found!', 404

    return Response(img.img, mimetype=img.mimetype)'''

if __name__=="__main__":
    print("Starting Flask Server for Image Processing")
    util.load_saved_artifacts()
    app.run(port=5000)