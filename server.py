from flask import Flask, request, jsonify

app = Flask(__name__)

weather_data = {}

@app.route('/endpoint', methods=['GET', 'POST'])
def endpoint():
    global weather_data
    if request.method == 'POST':
        data = request.json
        if data:
            weather_data = data
            return jsonify({"status": "success"}), 200
        else:
            return jsonify({"error": "No data received"}), 400
    elif request.method == 'GET':
        return jsonify(weather_data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
