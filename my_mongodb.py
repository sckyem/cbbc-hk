from default_modules import *

class Mongodb:

    def __init__(self, collection='test', document='test'):
        self.COLLECTION = collection
        self.DOCUMENT = document

    def atlas_conn(self, is_ping=False):
        uri = ""
        try:
            conn = MongoClient(uri, server_api=ServerApi('1'))
            if is_ping:
                conn.admin.command('ping')
                print("Pinged your deployment. You successfully connected to MongoDB!")
            return conn
        except Exception as e:
            print(e)

    def conn(self):
        return self.atlas_conn()

    def document(self):
        conn = self.conn()     
        return conn[self.COLLECTION][self.DOCUMENT]

    def format(self, dataframe):
        df = dataframe.to_frame() if isinstance(dataframe, pd.Series) else dataframe.copy()
        if isinstance(df, pd.DataFrame):
            df.columns = columns_to_strings(df.columns)
            df['_id'] = df.index
            return df.to_dict('records') 
        else:
            return df

    def insert(self, data, batch_size=1000):
        document = self.document()
        data = self.format(data)
        if isinstance(data, dict):
            result = document.insert_one(data)
            print("Inserting data Success" if result.acknowledged else "Error.")

        elif isinstance(data, list):
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                result = document.insert_many(batch)
                print("Inserting data Success" if result.acknowledged else "Error.")                
        else: print(f"{type(data)} cant insert" )

    def find(self, query={}, projection={}, is_dataframe=False):
        document = self.document()
        result = list(document.find(query, projection))
        if result and is_dataframe:
            df = pd.DataFrame(result)
            df = df.set_index(df.columns[0])
            df.columns = strings_to_columns(df.columns)
            return df
        else:
            return result

    def update(self, data):
        new = data.to_frame() if isinstance(data, pd.Series) else data.copy()
        if isinstance(new, pd.DataFrame):
            old = self.find({'_id': {'$gte': new.index[0], '$lte': new.index[-1]}}, is_dataframe=True)
            if old:
                if (list(old.columns) == list(new.columns)) and (type(old.index) == type(new.index)):
                    result = new.ne(old).any(axis=1)
                    diff = new.loc[result[result].index]
                    if not diff.empty:                    
                        document = self.document() 
                        diff.columns = columns_to_strings(diff.columns)
                        for index, row in diff.iterrows():
                            filter = {'_id': index}
                            update = {'$set': row.to_dict()}
                            result = document.update_one(filter, update, upsert=True)
                        print("Update Success" if result.acknowledged else "Error.")
                        return True
                    else:
                        print(f"No need to update.")
                        return False
            else:
                if not self.find():
                    self.insert(new)
                    return True
                else:
                    print("Update error")

    def read(self, query={}, projection={}, is_dataframe=False):
        return self.find(query, projection, is_dataframe)

    def write(self, data):
        return self.update(data)
