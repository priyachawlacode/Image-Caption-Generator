from flask import Flask,render_template,request,redirect, url_for
from werkzeug.utils import secure_filename
from predict import predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/services', methods = ['GET', 'POST'])
def services():
	if request.method == 'POST':
		ans="\""
		f = request.files['file']
		f.filename="user_img.jpg"
		f.save("static/User_Images/"+f.filename)
		ans=ans+predict("static/User_Images/user_img.jpg").title()+"\""
		print("ANSWWRRRRR ISS",ans)
		return render_template("services.html",ans=ans)
	return render_template("services.html")

@app.route('/single')
def single():
	return render_template("single.html")


if __name__ == '__main__':
    app.run(host="localhost", port=7000, debug=True)