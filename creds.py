from pymongo import MongoClient

pw = "password"
uri = f"mongodb://cwward:{pw}@wranglerdb01a.uvm.edu:27017/?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"
client = MongoClient(uri)

s2orc_token = '8mH99xWXoi60vMDfkSJtb6zVhSLiSgNP8ewg3nlZ'