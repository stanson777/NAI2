import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

base_dir=os.path.dirname(os.path.abspath(__file__))
data_dir=os.path.join(base_dir,'nba')

file_paths = []
for dirname, _, filenames in os.walk(data_dir):
    for filename in filenames:
        full_path = os.path.join(dirname, filename)
        file_paths.append(full_path)


data_frames = [pd.read_csv(file) for file in file_paths]
combined_df = pd.concat(data_frames, ignore_index=True)


combined_df = combined_df.drop_duplicates()
print(combined_df.isnull().sum())


features = ['SHOT_DISTANCE', 'SHOT_TYPE', 'ZONE_RANGE', 'BASIC_ZONE', 'LOC_X', 'LOC_Y', 'QUARTER', 'MINS_LEFT', 'SECS_LEFT']
y = combined_df['SHOT_MADE']
X = combined_df[features]


label_columns = ['SHOT_TYPE', 'ZONE_RANGE', 'BASIC_ZONE']
X_encoded = X.copy()

for col in label_columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])


train_X, test_X, train_y, test_y = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


nba_shoots_modelrf = DecisionTreeClassifier(random_state=42)
nba_shoots_modelrf.fit(train_X, train_y)


test_predictions = nba_shoots_modelrf.predict(test_X)
accuracy = accuracy_score(test_y, test_predictions)
print(f"Dokładność na zbiorze testowym: {accuracy * 100:.2f}%")


def predict_shot(shoot_distance, shot_type, zone_range, basic_zone, loc_x, loc_y, quarter, mins_left, secs_left):
    input_data = np.array([[shoot_distance, shot_type, zone_range, basic_zone, loc_x, loc_y, quarter, mins_left, secs_left]])
    input_df = pd.DataFrame(input_data, columns=features)


    for col in label_columns:
        input_df[col] = le.transform(input_df[col])

    prediction = nba_shoots_modelrf.predict(input_df)
    return "Rzut się udał!" if prediction[0] == 1 else "Rzut się nie udał!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict NBA shot outcome.')
    parser.add_argument('shoot_distance', type=float, help='Distance of the shot')
    parser.add_argument('shot_type', type=str, help='Type of the shot (e.g., "Jump Shot", "Layup", etc.)')
    parser.add_argument('zone_range', type=str, help='Zone range (e.g., "Restricted Area", "Mid-Range", etc.)')
    parser.add_argument('basic_zone', type=str, help='Basic zone (e.g., "Left Side", "Right Side", etc.)')
    parser.add_argument('loc_x', type=float, help='X-coordinate location of the shot')
    parser.add_argument('loc_y', type=float, help='Y-coordinate location of the shot')
    parser.add_argument('quarter', type=int, help='Quarter of the game (1-4)')
    parser.add_argument('mins_left', type=int, help='Minutes left in the quarter')
    parser.add_argument('secs_left', type=int, help='Seconds left in the quarter')

    args = parser.parse_args()


    result = predict_shot(args.shoot_distance, args.shot_type, args.zone_range, args.basic_zone,
                          args.loc_x, args.loc_y, args.quarter, args.mins_left, args.secs_left)

    print(result)
