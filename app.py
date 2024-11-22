from flask import Flask, request, jsonify,render_template
import pickle
import pandas as pd
import random

app = Flask('mental-health')
from health.utils import get_options, get_gpt_response, generate_prompt

# Load the model
model_path = './models/catboost_(students)_model.pkl'  # Update with your model path
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/routes')
def list_routes():
    import urllib
    from flask import url_for
    output = []
    for rule in app.url_map.iter_rules():
        options = {}
        for arg in rule.arguments:
            options[arg] = "[{}]".format(arg)
        methods = ','.join(rule.methods)
        url = url_for(rule.endpoint, **options)
        line = urllib.parse.unquote("{:50s} {:20s} {}".format(rule.endpoint, methods, url))
        output.append(line)
    return "<br>".join(output)



@app.route('/api/options')
def api_options():   
    return jsonify(get_options())



@app.route('/api/assess',methods=['POST'])
def api_assess():   
    data = request.get_json(force=True)
    happy_user = [
        "You're doing great! It's wonderful to see that you're in a positive mental space—keep nurturing your well-being and embracing your strengths.",
        "This is a fantastic milestone! Staying in tune with your mental health is a powerful act of self-care, and you're doing it beautifully.",
        "Your assessment results reflect your resilience and balance. Keep investing in yourself and your happiness!",
        "Great job prioritizing your mental health! Staying proactive about it shows your commitment to living your best life.",
        "Your mental wellness is an inspiration! Keep building on these positive habits and spreading the good vibes around you."
    ]
    # input_data = pd.DataFrame([data])
    # prediction = model.predict(input_data)  
    print('new input data',data)
    content = random.choice(happy_user)
    prediction = 0
    prompt = ''
    try :
        res = generate_prompt(data) 
        prediction = res['prediction']
        prompt = res['prompt']
        if prediction == 0 :
            content = 'Keep up the good work, You are looking good '+ content
        else: 
            content = get_gpt_response(prompt)
    except Exception as err:
        content = 'An error occured please try again, make sure you have filled all fields'
        print("API Error",err)
    
    return jsonify({
        'prediction':prediction,
        'prompt':prompt,
        'content' :content
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True,port=5001)
