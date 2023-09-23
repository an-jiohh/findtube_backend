from flask import Flask
import config

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!' + config.test


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5655)
