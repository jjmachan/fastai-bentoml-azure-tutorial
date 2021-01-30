import shutil
import os
import bentoml
from fastai.vision.all import *

from bentoservice import DogVCatService


def label_func(f): return str(f)[0].isupper()


bento_folder = 'bento_service'
shutil.rmtree(bento_folder)
os.makedirs(bento_folder)
learner = load_learner('model.pkl')
svc = DogVCatService()
svc.pack('learner', learner)
svc.save()
