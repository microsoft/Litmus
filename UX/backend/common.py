import os
import json

import azure.functions as func

# Allowing CORS
RESPONSE_HEADERS = {}

"""
    Shorthand for sending JSON responses
"""
def jsonResponse(jsonDict, status_code=200, headers=RESPONSE_HEADERS, mimetype="application/json"):
    return func.HttpResponse(json.dumps(jsonDict), status_code=status_code, headers=headers, mimetype=mimetype)


"""
    Shorthand for serving files
"""
def fileResponse(filePath, status_code=200, headers=RESPONSE_HEADERS, mimetype="text/html"):
    if os.path.exists(filePath):
        with open(filePath, "rb") as fin:
            fileData = fin.read()
        return func.HttpResponse(fileData, status_code=status_code, headers=RESPONSE_HEADERS, mimetype=mimetype)
    return None

    
"""
    Dumps binary data to file
"""
def dumpBinaryToFile(dirpath, filname, data):
    filPath = os.path.join(dirpath, filname)
    with open(filPath, "wb") as fout:
        fout.write(data)
    return filPath