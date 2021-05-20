import argparse
from typing import *
import numpy as np

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import server.api as api
import path_fixes as pf

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--port", default=8000, type=int, help="Port to run the app. ")

app = FastAPI()

# Allows the communication of frontend and backend that run from different origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Main routes
@app.get("/")
def index():
    """For local development, serve the index.html in the dist folder"""
    return RedirectResponse(url="client/index.html")

# the `file_path:path` says to accept any path as a string here. Otherwise, `file_paths` containing `/` will not be served properly
@app.get("/client/{file_path:path}")
def send_static_client(file_path:str):
    """ Serves (makes accessible) all files from ./client/ to ``/client/{path}``. Used primarily for development. NGINX handles production.

    Args:
        path: Name of file in the client directory
    """
    f = str(pf.DIST / file_path)
    print("Finding file: ", f)
    return FileResponse(f)

# ======================================================================
## MAIN API - Backend server endpoint ##
# ======================================================================

# GET to read/create simple data. Parameters that are not part of the path automatically become query parameters.
@app.get("/api/get-a-hi", response_model=str)
async def hello(firstname:str, age:int=45):
    return "Hello " + firstname

# POST to send/ create Object data, response_model converts output data to its type declaration.
@app.post("/api/post-a-bye", response_model=str)
async def goodbye(payload:api.GoodbyePayload):
    # Coerce into correct type. Not needed if no test written for this endpoint
    payload = api.GoodbyePayload(**payload)
    return "Goodbye " + payload.firstname

if __name__ == "__main__":
    # This file is not run as __main__ in the uvicorn environment
    args, _ = parser.parse_known_args()
    uvicorn.run("server:app", host='127.0.0.1', port=args.port)
