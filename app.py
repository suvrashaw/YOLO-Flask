from flask import Flask, render_template, request

app = Flask(__name__, static_folder='static')

@app.route('/')
def web():
    return render_template('web.html')

if __name__ == "__main__":
    app.run(debug=True)