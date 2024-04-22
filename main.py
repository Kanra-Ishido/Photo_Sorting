from fastapi import FastAPI, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from fastapi.responses import HTMLResponse, FileResponse
from inference import predict
from typing import List
import uvicorn
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "static/Uploaded_images"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
def read_root(request: Request):
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def upload_files(request: Request, category: str = Form(...), files: List[UploadFile] = File(...)):
    all_probability = []
    images_path_list = []
    for file in files:
        input_image_path = os.path.join(UPLOAD_DIR, file.filename)
        os.makedirs(os.path.dirname(input_image_path), exist_ok=True)
        with open(input_image_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)

        result_text = predict(input_image_path)
        probability = []
        food = result_text["Food"]
        friend = result_text["Friend"]
        id_photo = result_text["ID_Photo"]
        scenery= result_text["Scenery"]
        study = result_text["Study"]

        probability.append(food)
        probability.append(friend)
        probability.append(id_photo)
        probability.append(scenery)
        probability.append(study)
        all_probability.append(probability)

        image_url = f"{input_image_path}"
        images_path_list.append(image_url)
    
    paired = list(zip(images_path_list, all_probability))
    if category == "food":
        sorted_pairs = sorted(paired, key=lambda x: x[1][0], reverse=True)
    elif category == "friend":
        sorted_pairs = sorted(paired, key=lambda x: x[1][1], reverse=True)
    elif category == "id_photo":
        sorted_pairs = sorted(paired, key=lambda x: x[1][2], reverse=True)
    elif category == "scenery":
        sorted_pairs = sorted(paired, key=lambda x: x[1][3], reverse=True)
    elif category == "study":
        sorted_pairs = sorted(paired, key=lambda x: x[1][4], reverse=True)

    sorted_photo_names_by_food = [pair[0] for pair in sorted_pairs]
    first_photo = sorted_photo_names_by_food[0]
    second_photo = sorted_photo_names_by_food[1]
    third_photo = sorted_photo_names_by_food[2]
    fourth_photo = sorted_photo_names_by_food[3]
    fifth_photo = sorted_photo_names_by_food[4]
    
    return templates.TemplateResponse("predict.html", 
                                      {"request": request, 
                                       "category" : category,
                                       "first_photo": first_photo,
                                       "second_photo": second_photo,
                                       "third_photo": third_photo,
                                       "fourth_photo": fourth_photo,
                                       "fifth_photo": fifth_photo,
                                       })

@app.get("/predict.html")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
