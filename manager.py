from notebook.services.contents.filemanager import FileContentsManager
from notebook.services.contents.filecheckpoints import GenericFileCheckpoints, FileManagerMixin

from datetime import datetime
import nbformat as nbf
import pathlib
import shutil
import os
from .nbexplode import explode, recombine


class NBExplodeManager(FileContentsManager):
    
    def _notebook_model(self, model, path):
        path = path.strip('/')
        stats = os.lstat(path + ".exploded")
 
        nb_model = {}
        nb_model["created"] = datetime.utcfromtimestamp(stats.st_ctime) 
        nb_model["last_modified"] = datetime.utcfromtimestamp(stats.st_mtime)
        nb_model["mimetype"] = None
        nb_model["type"] = "notebook"
        nb_model["name"] = path.rsplit("/")[-1]
        nb_model["path"] = path
        try:
            open(path)
        except OSError as e:
            nb_model["writable"] = True
        else:
            nb_model["writable"] = True
        return nb_model


    def save(self, model, path=''):
        if (model["type"] == "notebook" or path.endswith(".ipynb")):
            path = path.strip('/')
            nb = nbf.from_dict(model["content"])
            directory = pathlib.Path(path + '.exploded')
            if directory.exists():
                shutil.rmtree(path + '.exploded')
            directory.mkdir()
            self.log.info(model)
            explode(nb, directory)
            return self.get(path, content=False)
            
        return super(NBExplodeManager, self).save(model, path)

    def get(self, path, content=True, type=None, format=None):
        path = path.strip("/")
        if type == "notebook" or path.endswith("ipynb"):
            new_path = path + ".exploded"
            directory = pathlib.Path(new_path)
            merged_nb = recombine(directory)
            model = self._notebook_model(merged_nb, path)
            model["content"] = merged_nb if content else None
            model["format"] = "json" if content else None
            return model

        elif path.endswith(".exploded"): # for the /tree handler
            return self.get(path.strip(".exploded"), content=content)

        return super(NBExplodeManager, self).get(path, content, type, format)

    def file_exists(self, path):

        if path.endswith(".ipynb"):
            path = path.strip("/")
            path = self._get_os_path(path)
            return os.path.isdir(path + ".exploded")
        return super(NBExplodeManager, self).file_exists(path)


    def exists(self, path):
        return self.file_exists(path) or self.dir_exists(path)


class NBExplodingCheckPoints(GenericFileCheckpoints, FileManagerMixin):

    def create_notebook_checkpoint(self, nb, path):
        (basedir, name) = path.rsplit("/", 1)
        basedir = pathlib.Path(basedir.strip("/"))
        new_checkpoints_path = basedir / ".ipynb_checkpoints" / (name + ".exploded")
        if not new_checkpoints_path.exists():
            new_checkpoints_path.mkdir()
        explode(nbf.from_dict(nb), new_checkpoints_path)

        return self.checkpoint_model("checkpoint", new_checkpoints_path.name)
    


    

    
        
        
        





    