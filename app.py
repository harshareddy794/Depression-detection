from flask import Flask, request,render_template
import numpy as np
import pickle

reg = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__,template_folder='templates')

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/clasify",methods=["POST"])
def clasify():
    # print(request.form)
    sm = float(request.form['sm'])
    person = float(request.form['person'])
    issue = float(request.form['issues'])
    sc = float(request.form['sc'])
    life = float(request.form['life'])
    depression = float(request.form['depression'])
    arr = np.array([[sm,person,issue,sc,life,depression]])
    predection = reg.predict(arr)
    return render_template("clasify.html",data = predection)
    return "hello"

if __name__ == "__main__":
    app.run(debug=True)