import joblib
from sklearn.ensemble import RandomForestClassifier

def split_random_forest_model(model_path, output_directory):
    # Load the original Random Forest model
    original_model = joblib.load(model_path)
    
    # Access individual decision trees
    individual_trees = original_model.estimators_
    
    # Save each tree as a separate file in the output directory
    for i, tree in enumerate(individual_trees):
        tree_filename = f'{output_directory}/decision_tree_{i}.pkl'
        joblib.dump(tree, tree_filename)
    
    print(f'Split {len(individual_trees)} trees and saved them in the "{output_directory}" directory.')

if __name__ == "__main__":
    model_path = 'models/hotel_cancellation_model.pkl'  
    output_directory = 'models/individual_trees' 
    
    split_random_forest_model(model_path, output_directory)
