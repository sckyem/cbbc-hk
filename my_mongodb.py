from pymongo.mongo_client import MongoClient
#from pymongo.server_api import ServerApi
from default_modules import *

def list_database_names(host_port='192.168.1.59:27017'):
    client = MongoClient(f"mongodb://{host_port}")
    return client.list_database_names()

def list_collection_names(collection_name='test', host_port='192.168.1.59:27017'):
    client = MongoClient(f"mongodb://{host_port}")
    return client[collection_name].list_collection_names()

class Mongodb:

    def __init__(self, collection_name='test', document_name='test', host_port='192.168.1.59:27017'):
        self.HOST_PORT = host_port
        self.COLLECTION_NAME = collection_name
        self.DOCUMENT_NAME = document_name
        self.client = MongoClient(f"mongodb://{self.HOST_PORT}")
        self.document = self.client[self.COLLECTION_NAME][self.DOCUMENT_NAME]

    # def atlas(self, is_ping=False):
    #     self.secret = st.secrets['mongodb_pw']
    #     uri = f"mongodb+srv://{self.secret}@cluster0.xovoill.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    #     try:
    #         conn = MongoClient(uri, server_api=ServerApi('1'))
    #         if is_ping:
    #             conn.admin.command('ping')
    #             print("Pinged your deployment. You successfully connected to MongoDB!")
    #         return conn
    #     except Exception as e:
    #         print(e)

    def documents(self):
        return self.client.list_database_names()

    def delete(self):
        """
        Deletes all documents from the collection.
        """
        self.document.delete_many({})
        print(f"Deleted {self.COLLECTION_NAME} {self.DOCUMENT_NAME} documents.")

    def insert(self, dataframe, batch_size=1000):
        df = dataframe.copy()
        if isinstance(dataframe, pd.Series):
            df = dataframe.to_frame() 

        if isinstance(df, pd.DataFrame):
            df.columns = columns_to_strings(df.columns)
            df['_id'] = df.index
            data = df.to_dict('records')
        else:
            data = df

        if isinstance(data, dict):
            result = self.document.insert_one(data)
            print(f"Inserting data {self.COLLECTION_NAME} {self.DOCUMENT_NAME}" + 'success' if result.acknowledged else "Error.")

        elif isinstance(data, list):
            for idx in range(0, len(data), batch_size):
                batch = data[idx:idx+batch_size]
                result = self.document.insert_many(batch)
                print(  f"Inserting {self.COLLECTION_NAME} {self.DOCUMENT_NAME} " + "Success." if result.acknowledged else "Error.")                
        else: print(f"{type(data)} cant insert" )

    def find(self, query={}, projection={}, is_dataframe=False):
        result = list(self.document.find(query, projection))
        if result:
            if is_dataframe:
                df = pd.DataFrame(result)
                df = df.set_index(df.columns[0])
                return df
            else:
                return result

    def update(self, data, allow_replace=False):
        new = data.to_frame() if isinstance(data, pd.Series) else data.copy()
        if isinstance(new, pd.DataFrame) and not new.empty:
            if isinstance(new.columns, pd.MultiIndex):
                new.columns = columns_to_strings(new.columns)
            old = self.find({'_id': {'$gte': new.index[0], '$lte': new.index[-1]}}, is_dataframe=True)             
            if isinstance(old, pd.DataFrame):
                if is_structure_same(old, new):
                    mask = new.ne(old).any(axis=1)
                    update_df = new.loc[new.index.isin(mask[mask].index)]
                    if not update_df.empty:
                        for index, row in update_df.iterrows():
                            filter = {'_id': index}
                            update = {'$set': row.to_dict()}
                            result = self.document.update_one(filter, update, upsert=True)
                        print(  f"Update " + f"{'success' if result.acknowledged else 'error'}" + f" for {self.COLLECTION_NAME} {self.DOCUMENT_NAME}"  )
                        return True
                    else:
                        print(  f"No update for {self.COLLECTION_NAME} {self.DOCUMENT_NAME}."  )
                        return False
                else:
                    if allow_replace:
                        self.delete()
                        self.insert(new)
                        return True
            else:
                if not self.find():
                    self.insert(new)
                    return True
                else:
                    print(  f"Update error for {self.COLLECTION_NAME} {self.DOCUMENT_NAME}"  )

    def read(self, query={}, projection={}, is_dataframe=False):
        return self.find(query, projection, is_dataframe)

    def write(self, data):
        return self.update(data)
    
    def last_id(self):
        last_document = self.document.find_one(sort=[('_id', -1)])
        if isinstance(last_document, dict) and '_id' in last_document:
            return last_document['_id']
    
    def first_id(self):
        return self.document.find_one()['_id']
    
if __name__ == '__main__':

    pass