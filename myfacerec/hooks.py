# myfacerec/hooks.py

from typing import Callable, List, Optional

class Hooks:
    def __init__(self):
        self.before_detect: Optional[List[Callable]] = []
        self.after_detect: Optional[List[Callable]] = []
        self.before_embed: Optional[List[Callable]] = []
        self.after_embed: Optional[List[Callable]] = []

    def register_before_detect(self, func: Callable):
        self.before_detect.append(func)

    def register_after_detect(self, func: Callable):
        self.after_detect.append(func)

    def register_before_embed(self, func: Callable):
        self.before_embed.append(func)

    def register_after_embed(self, func: Callable):
        self.after_embed.append(func)

    def execute_before_detect(self, image):
        for func in self.before_detect:
            image = func(image)
        return image

    def execute_after_detect(self, boxes):
        for func in self.after_detect:
            func(boxes)

    def execute_before_embed(self, image, boxes):
        for func in self.before_embed:
            func(image, boxes)

    def execute_after_embed(self, embeddings):
        for func in self.after_embed:
            func(embeddings)
