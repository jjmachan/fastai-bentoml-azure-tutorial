from bentoml.frameworks.fastai import FastaiModelArtifact
from bentoml.adapters import FileInput
from fastcore.utils import tuplify, detuplify

import bentoml
import datablock_utils


@bentoml.artifacts([FastaiModelArtifact('learner')])
@bentoml.env(infer_pip_packages=True)
class DogVCatService(bentoml.BentoService):

    @bentoml.api(input=FileInput(), batch=True)
    def predict(self, files):
        files = [i.read() for i in files]
        dl = self.artifacts.learner.dls.test_dl(files, rm_type_tfms=None, num_workers=0)
        inp, preds, _, dec_preds = self.artifacts.learner.get_preds(dl=dl, with_input=True, with_decoded=True)

        return [bool(i) for i in dec_preds]
