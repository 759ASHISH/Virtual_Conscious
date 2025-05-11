from flask import Flask, render_template, request, jsonify
from chat import chat_with_memory

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    response = chat_with_memory(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
