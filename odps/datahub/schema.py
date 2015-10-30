import json
from odps.models import Schema

class DatahubSchema(Schema):
    def __init__(self, root):
        node = root.get('columns')
        columns = json.loads(node)
        for col in columns:
            column = Column()
            for key in col:
                setattr(column, key, item[key])
            add_column(column)
            
