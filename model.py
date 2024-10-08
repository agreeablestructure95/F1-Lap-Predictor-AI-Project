import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

files = ['formula_one_data_A.csv', 'formula_one_data_B.csv', 'formula_one_data_C.csv', 'formula_one_data_D.csv']
df_list = [pd.read_csv(file) for file in files]
df = pd.concat(df_list, ignore_index=True)

categorical_columns = ['car_tire_type', 'prev_track', 'prev_climatic_conditions', 'current_tire_type']

df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
features = ['driver_awareness', 'driver_pace', 'car_engine_power', 'car_fuel_efficiency',
             'amount_of_fuel', 'weightof_car'] + \
            [col for col in df.columns if col.startswith(tuple(categorical_columns))]

X = df[features]
y = df['prev_lap_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

def convert_time(minutes):
    total_seconds = round(minutes * 60) 
    minutes_int = total_seconds // 60   
    seconds_int = total_seconds % 60  
    milliseconds = round((minutes * 60 - total_seconds) * 100) 
    return f"{minutes_int:02}:{seconds_int:02}:{milliseconds:02}"


def predict_lap_time(driver_id, current_tire_type, current_track):
    if driver_id not in df['driver_id'].values:
        return "Driver ID not found in the dataset."
    
    driver_data = df[df['driver_id'] == driver_id].iloc[0]

    features_df = pd.DataFrame({
        'driver_awareness': [driver_data['driver_awareness']],
        'driver_pace': [driver_data['driver_pace']],
        'car_engine_power': [driver_data['car_engine_power']],
        'car_fuel_efficiency': [driver_data['car_fuel_efficiency']],
        'amount_of_fuel': [driver_data['amount_of_fuel']],
        'weightof_car': [driver_data['weightof_car']]
    })
    

    for col in [col for col in df.columns if col.startswith(tuple(categorical_columns))]:
        features_df[col] = 0
    
    features_df[f'current_tire_type_{current_tire_type}'] = 1
    features_df[f'prev_track_{current_track}'] = 1
    

    for col in [col for col in df.columns if col.startswith(tuple(categorical_columns))]:
        if col not in features_df.columns:
            features_df[col] = 0
    
  
    features_df = features_df.reindex(columns=X.columns, fill_value=0)
    

    results = {}
    tire_types = ['Soft', 'Medium', 'Hard']
    
    for tire in tire_types:
        features_df_temp = features_df.copy()
        features_df_temp[f'car_tire_type_{tire}'] = 1
        for other_tire in tire_types:
            if other_tire != tire:
                features_df_temp[f'car_tire_type_{other_tire}'] = 0
                

        features_df_temp = features_df_temp.reindex(columns=X.columns, fill_value=0)
        
        # Predict lap time
        predicted_time = model.predict(features_df_temp)[0]
        results[f"{tire}"] = convert_time(predicted_time)
    
    return results


driver_id = int(input("Enter the driver ID to predict lap time: "))
current_tire_type = input("Enter the current tire type (Soft, Medium, Hard): ")
current_track = input("Enter the current track (Track A, Track B, Track C, Track D): ")


predictions = predict_lap_time(driver_id, current_tire_type, current_track)


print(f"Lap Time Predictions for Different Tire Types:")
for tire, time in predictions.items():
    print(f"{tire}: {time}")

best_tire = min(predictions, key=lambda k: int(predictions[k].split(':')[0]) * 60 + int(predictions[k].split(':')[1]) + int(predictions[k].split(':')[2]) / 100)
print(f"\nBest tire type based on predictions: {best_tire}")
print(f"Suggested changes: Consider changing to {best_tire} tires for optimal performance.")
