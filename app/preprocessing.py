import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def prepare_data():
    data = {
        'text': [
            'Free money!!!', 
            'Call this number for a prize!', 
            'Meeting at 10 am tomorrow', 
            'Your invoice is due next week', 
            'Congratulations, you won a gift card!'
        ],
        'labels': ['spam', 'spam', 'not spam', 'not spam', 'spam']
    }
    return pd.DataFrame(data)
