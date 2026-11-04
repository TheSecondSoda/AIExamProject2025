import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, make_response
import os
from DataLoader import PetFinderDataset

#Setup flask api
app = Flask(__name__)
port = int(os.environ.get('PORT', 5000))

#Load the dataset
file_path = "Part2Petfinder/petfinder-pawpularity-score/train.csv"
df = pd.read_csv(file_path)

#Clean the data
df.drop(columns=['Id'], inplace=True)

X = df.drop(columns=['Pawpularity'])
y = df['Pawpularity']

#Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Fit the model
reg = LinearRegression()
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

#Make and print a results table
results_table = pd.DataFrame(data=X_test, columns=['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'])
results_table['Predicted Score'] = y_pred
results_table['Actual Score'] = y_test
results_table['Error'] = results_table['Actual Score']-results_table['Predicted Score']
results_table['Error Squared'] = results_table['Error']* results_table['Error']

print(results_table)


#Make flask api endpoint
@app.route("/", methods=['GET'])
def predict_CR(*args):
    #Get arguments
    SubjectFocus = request.args.get("Subject Focus")
    Eyes = request.args.get("Eyes")
    Face = request.args.get("Face")
    Near = request.args.get("Near")
    Action = request.args.get("Action")
    Accessory = request.args.get("Accessory")
    Group = request.args.get("Group")
    Collage = request.args.get("Collage")
    Human = request.args.get("Human")
    Occlusion = request.args.get("Occlusion")
    Info = request.args.get("Info")
    Blur = request.args.get("Blur")

    #Input validation, all inputs must be convertable into floats
    invalid_arguments = []
    for argument in [[SubjectFocus, "Subject Focus"], [Eyes, "Eyes"], [Face, "Face"], [Near, "Near"], [Action, "Action"], [Accessory, "Accessory"], [Group, "Group"], [Collage, "Collage"], [Human, "Human"], [Occlusion, "Occlusion"], [Info, "Info"], [Blur, "Blur"]]:

        if argument[0] == None:
            invalid_arguments.append(argument[1])
            continue
        try:
            float(argument[0])
        except ValueError:
            invalid_arguments.append(argument[1])

    if len(invalid_arguments) > 0:
        return jsonify(f"Invalid input, the following are not numbers: {invalid_arguments}"), 403

    #Predict a pawpularity score
    data = []
    data.append( {"Subject Focus": SubjectFocus, "Eyes": Eyes, "Face": Face, "Near": Near, "Action": Action, "Accessory": Accessory, "Group": Group, "Collage": Collage, "Human": Human, "Occlusion": Occlusion, "Info": Info, "Blur": Blur})
    df2 = pd.DataFrame(data)

    prediction = reg.predict(df2).item()
    return jsonify(prediction), 200

#Run flask api
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)