import pickle

model,encoders = pickle.load(open("accident_model.pkl","rb"))

place="Hyderabad"
weather="Rainy"
traffic="High"
speed=82

p=encoders["Place"].transform([place])[0]
w=encoders["Weather"].transform([weather])[0]
t=encoders["Traffic"].transform([traffic])[0]

prediction=model.predict([[p,w,t,speed]])

risk=encoders["Risk"].inverse_transform(prediction)[0]

print("Predicted Risk =",risk)