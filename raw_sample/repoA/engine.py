import os
import sys
import json


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, items):
        out = []
        for it in items:
            if it > 0:
                for j in range(it):
                    if j % 2 == 0:
                        out.append(j)
                    else:
                        try:
                            out.append(1 / j)
                        except ZeroDivisionError:
                            out.append(0)
        return out
